import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.translate import AlignedSent, Alignment, IBMModel2
import os
import json
import pdb
import copy
from nltk.probability import (FreqDist,
    ConditionalProbDist,
    ConditionalFreqDist,
    LidstoneProbDist)
from nltk.translate.ibm_model import AlignmentInfo
import sys
import io
from tqdm import tqdm
import itertools
import math


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

ALPHA = 0.015
ZERO = 1e-20

    
def p_product_LM(fr_test_text):
    global en_vocab
    global en_cpd
    global en_cfd
    global fd
    product_prob_dict = nltk.FreqDist({"=START=":1})

    # Enumerate all the combinations of possible translation
    # Complexity is v^3  v:vocabulary of english tokens
    # As we omit the extreme small probabilities, we speed up the process.
    for amot in fr_test_text:
        aword_1 = ""
        product_prob_dictn_new = nltk.FreqDist()
        for en_sent in product_prob_dict:
            aword_1 = nltk.word_tokenize(en_sent)[-1]
                    
            for aword in en_vocab:
                # Probability = Language Model * Translate Model
                if aword in en_cpd[aword_1]:
                    p_lm = en_cpd[aword_1][aword]
                else:
                    p_lm = smooth(aword_1,en_cfd)
                # p(e|f) => p(f|e) * p(e)
                # p(f|e) translation model
                # p(e) language model
                # fd[aword][amot]:    p(f|e) 
                # p_lm:                 p(e)
                cur_prob = p_lm*fd[aword][amot]
                
                product = product_prob_dict[en_sent]
                # If probability is too small, ignore this translation.
                # To save the space and processing time.
                # If the ZERO is too big, the correct translation will be ignored.
                if product*cur_prob>ZERO:
                    if en_sent=="=START=":
                        product_prob_dictn_new[aword] = product*cur_prob
                    else:
                        product_prob_dictn_new[en_sent+" "+aword] = product*cur_prob
        product_prob_dict = product_prob_dictn_new
    return product_prob_dict



# Laplace Smooth
def cal_laplace(cfd):
    cpd = {}
    global ALPHA
    for c in cfd:
        cpd[c]={}
        for fd in cfd[c]:
            # Estimator theta_i = (x_i+alpha)/(N+alpha*d)  i = 1...d
            cpd[c][fd]= (cfd[c][fd]+ALPHA)/(cfd[c].N()+ALPHA*len(en_vocab))
    return cpd
    
def smooth(c,cfd):
    global ALPHA
    return ALPHA/(cfd[c].N()+ALPHA*len(en_vocab))

cwd = os.getcwd()
nltk.data.path.append(cwd)


# translation_table[trg_word][src_word]
# alignment_table[i][j][l][m]
# Notations:
# i: Position in the source sentence
#     Valid values are 0 (for NULL), 1, 2, ..., length of source sentence
# j: Position in the target sentence
#     Valid values are 1, 2, ..., length of target sentence
# l: Number of words in the source sentence, excluding NULL
# m: Number of words in the target sentence
# s: A word in the source language
# t: A word in the target language
######
# bitext.append(AlignedSent(target_sentenses,source_sentenses))
# alignment: [(target_index,source_index),(target_index,source_index) ... ]
#                 target_index= 0,1,2,...,m-1 ; source_index= 0,1,2,...,l-1

def translate(fr_str):
    global fd
    global ibm2
    fr_test = fr_str
    fr_test_text = nltk.Text(nltk.word_tokenize(fr_test))
    
    for t in en_vocab:
        fd[t] = nltk.FreqDist(ibm2.translation_table[t])
    prod = p_product_LM(fr_test_text)
    if len(prod)>0:
        return prod.most_common(1)[0][0]
    else:
        return ""

def translate_ibm(fr_str): 
    global fd
    global ibm2
    fr_test = fr_str
    fr_test_text = nltk.Text(nltk.word_tokenize(fr_test))
    en_out = ""
   
    for f in fr_test_text:
      en_word = ""
      max_prob = 0 
      cur_prob = 0
      for e in en_vocab:
        if e in ibm2.translation_table:
          cur_prob = ibm2.translation_table[e][f]
          if cur_prob > max_prob:
            max_prob = cur_prob
            en_word = e
      en_out += " "+en_word
    return en_out 

def best_target_len(l):
  global cfd_len_l_m
  return cfd_len_l_m[l].most_common(1)[0][0]
 
# i = a(j)corpus_AT
# We want to find the best a(j) given j
def best_align_pos(j,l,m):
  global ibm2
  max_prob = 0
  cur_prob = 0
  i_pos = None
  for i in ibm2.alignment_table:
    cur_prob = ibm2.alignment_table[i][j][l][m]
    if cur_prob > max_prob:
      max_prob = cur_prob
      i_pos = i
  return i_pos

def translate_ibm2(fr_str): 
    global fd
    global ibm2
    
    fr_test = fr_str
    fr_test_text = nltk.Text(nltk.word_tokenize(fr_test))
    # Candidate English words list
    en_ibm1_text = nltk.Text(nltk.word_tokenize(translate_ibm(fr_test)))
    align_best = None
    en_out = ""
    
    max_prob = 0 
    cur_prob = 0
    # Choose target sentense length
    l = len(fr_test_text)
    #m = best_target_len(lcorpus_AT)
    m = len(en_ibm1_text)
    # Generate alignments
    alignment=[(-1,-1)]*m
    for j in range(0,m):
      alignment[j] =  (j,best_align_pos(j+1,l,m)-1)
    
    en_list = ['']*m
    # Reorder the English sentense based on align
    for j in range(1,m+1):
      i = alignment[j-1][1]
      if i == -1:
        en_list[j-1] = "[None]"
      else:
        en_list[j-1] =   en_ibm1_text.tokens[i]  
        
    en_out = ' '.join(en_list)
    return en_out
      
# Train
# Translate with ibm2 in noisy channel model
# p(e|f) => p(f|e) * p(e)
# p(f|e) translation model
# p(e) language model

path = 'corpora/abc/corpus_AT.txt'
train_file = open(path, encoding='utf-8')
lines = train_file.readlines()
i=0
bitext=[]
fr_text=[]
en_text=[]
while i < len(lines):
    if i % 2==0:
        # Will use the ibm model to calculate p(f|e)
        # With the noisy model, the translation direction is reversed!
        bitext.append(AlignedSent([t.lower() for t in nltk.word_tokenize(lines[i+1])]
                                ,[t.lower() for t in nltk.word_tokenize(lines[i])]))
        fr_text+=nltk.word_tokenize(lines[i])
        en_text+=nltk.word_tokenize(lines[i+1])
    i += 1
train_file.close()

fr_text = nltk.Text(fr_text)
en_text = nltk.Text(en_text)
en_vocab = [v for v in en_text.vocab()]
ibm2 = IBMModel2(bitext, len(bitext)*5)


cfd_len_l_m = {}
len_l_m_list = []
for t in bitext:
  l = len(t.mots)
  m = len(t.words)
  len_l_m_list += [(l, m)]
cfd_len_l_m = nltk.ConditionalFreqDist(len_l_m_list)

# cfd_dict = {}
# # Get the cfd for the alignment
# for t in bitext:
#   l = len(t.mots)
#   m = len(t.words)
#   cfd_dict[(l,m)] = nltk.ConditionalFreqDist(t.alignment)

# Language model
en_ngrams = nltk.ngrams(en_text,2,pad_left=True,left_pad_symbol="=START=")
en_cfd = nltk.ConditionalFreqDist(en_ngrams)
en_cpd = cal_laplace(en_cfd)
fd = {}

# Test
test_file = 'corpora/example.txt'
result=""
with open(test_file, encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        result += translate(line.lower())+"\n"
    

# Write the output
out_file = nltk.data.find('corpora/output.txt')
with open(out_file, "a") as f:
    f.write(result)
    
print(result)
pdb.set_trace()