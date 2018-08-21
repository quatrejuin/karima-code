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

ALPHA = 0.1
ZERO = 1e-10

    
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

# Train
# Translate with ibm2 in noisy channel model
# p(e|f) => p(f|e) * p(e)
# p(f|e) translation model
# p(e) language model

path = nltk.data.find('corpora/abc/corpus_AT.1.txt')
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
ibm2 = IBMModel2(bitext, len(bitext)-1)

# Language model
en_ngrams = nltk.ngrams(en_text,2,pad_left=True,left_pad_symbol="=START=")
en_cfd = nltk.ConditionalFreqDist(en_ngrams)
en_cpd = cal_laplace(en_cfd)
fd = {}


# Test
test_file = nltk.data.find('corpora/example.txt')
result=""
with open(test_file) as f:
    lines = f.readlines()
    for line in lines:
        result += translate(line.lower())+"\n"
    

# Write the output
out_file = nltk.data.find('corpora/output.txt')
with open(out_file, "a") as f:
    f.write(result)

    
