import glob
import sklearn.manifold
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec
import seaborn as sns


#reading the files using the glob module
text_names = glob.glob('*.txt')
print(text_names)


raw_corpus = ""
for book in text_names:
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
  
model = gensim.models.Word2Vec(data,min_count = 10, size = 300, window = 7)


model.wv.most_similar("Stark")
print("the cosine similarity between Stark and Winterfell is {}".format(model.wv.similarity("Stark","Winterfell")))


#vector matrix

vector_matrix = model.wv.syn0

#reducing the dimensinality

tsne = sklearn.manifold.TSNE(n_components = 2, random_state = 0)
vector_matrix_2d = tsne.fit_transform(vector_matrix)

vector_points = pd.DataFrame(
        [(word,co_ordinate[0],co_ordinate[1])
              for word,co_ordinate in [(word,vector_matrix_2d[model.wv.vocab[word].index])
                  for word in model.wv.vocab
                            ]
],
columns=['words','x','y']
)


#print(vector_points.head(10))
    
    
#plotting the data 
    
sns.set_context("poster")
vector_points.plot.scatter("x","y",figsize = (20,12))




