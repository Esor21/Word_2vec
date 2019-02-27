import glob
import re
import nltk
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec


#reading the files using the glob module
book_names = glob.glob('*.txt')
print(book_names)


raw_corpus = ""
for book in book_names:
    txt = open(book,'r',encoding ='utf8')
    txt1 = txt.read()   
    raw_corpus += txt1 
        
        
#print('total length of corpus{}'.format(len(raw_corpus))) 
 
 #replacing escape character with space   
raw_corpus = raw_corpus.replace("\n"," ")

data = []

for sent in sent_tokenize(raw_corpus):
    temp = []
    for w in word_tokenize(sent):
        temp.append(w)
    
    data.append(temp)
    
    
print(data[:15])    
  
#model = gensim.models.Word2Vec(data,min_count = 10, size = 300, window = 7)


#model.wv.most_similar("Stark")
#print("the cosine similarity between Stark and Winterfell is {}".format(model.wv.similarity("Stark","Winterfell")))










