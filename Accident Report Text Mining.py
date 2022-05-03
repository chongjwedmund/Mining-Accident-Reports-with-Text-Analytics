# topic modelling method 
import numpy as np
import pandas as pd

osha=pd.read_table('osha.txt',header=None,names = ["ID", "Title", "Description"])

#########################################################Preprocessing
import nltk
from nltk.corpus import stopwords
mystopwords=stopwords.words("English") + ['employee', 'employees', 'worker', 'workers', 'killing', 'killed', 'kill', 'kills', 'injured', 'injures', 'injuring', 'injure', 'sustain', 'sustains', 'sustained', 'sustaining', 'suffer', 'suffers', 'suffering', 'suffered']
WNlemma = nltk.WordNetLemmatizer()

def pre_process(text):
    tokens = nltk.word_tokenize(text)
    tokens=[ WNlemma.lemmatize(t.lower()) for t in tokens]
    tokens=[ t for t in tokens if t not in mystopwords]
    tokens = [ t for t in tokens if len(t) >= 3 ]
    return(tokens)

text = osha['Title']
toks = text.apply(pre_process)

# Use dictionary (built from corpus) to prepare a DTM (using frequency)
import logging
import gensim 
from gensim import corpora

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Filter off any words with document frequency less than 5, or appearing in more than 75% of documents
dictionary = corpora.Dictionary(toks)
dictionary.filter_extremes(no_below=5, no_above=0.75)

#dtm here is a list of lists, which is exactly a matrix
dtm = [dictionary.doc2bow(d) for d in toks]

lda = gensim.models.ldamodel.LdaModel(dtm, num_topics = 4, id2word = dictionary, passes=10,chunksize=32,random_state=10)

lda.show_topics(10) # 10 toks per topic in descending order

##Evaluate the coherence score of LDA models
'''
u_mass:prefer the model close to 0
c_v: [0,1], prefer bigger value
Do not fully rely on the coherence score
'''
from gensim.models.coherencemodel import CoherenceModel
cm_umass = CoherenceModel(lda,  dictionary=dictionary, corpus=dtm, coherence='u_mass')
cm_cv = CoherenceModel(lda,  dictionary=dictionary, texts=toks, coherence='c_v')
lda_umass = cm_umass.get_coherence()
lda_cv = cm_cv.get_coherence()
print(lda_umass)
print(lda_cv)


#pip install pyLDAvis
import pyLDAvis.gensim
import pickle 
import pyLDAvis
# Visualize the topics
pyLDAvis.enable_notebook()
LDAvis_prepared = pyLDAvis.gensim.prepare(lda, dtm, dictionary)
pyLDAvis.show(LDAvis_prepared)

dict = {0: 'fall, fracture, leg, ladder', 
        1: 'finger, burned, amputated, machine', 
        2: 'crushed, truck, tree, forklift', 
        3: 'electric shock, fire, explosion'}

##Note that different runs result in different but simillar results if random_state is not specified
##Label the topics based on representing "topic_words"

# Get the topic distribution of documents
doc_topics = lda.get_document_topics(dtm)

from operator import itemgetter

#Select the best topic (with highest score) for each document
top_topic = [ max(t, key=itemgetter(1))[0] for t in doc_topics ]

topics_perDoc = [ dict[t] for t in top_topic ]
print (topics_perDoc)

####################################### How many dos in each topic?
labels, counts = np.unique(topics_perDoc, return_counts=True)
print (labels)
print (counts)

# clustering method 

####################################### TDM
from sklearn.feature_extraction.text import TfidfVectorizer


# Create tfidf matrix
vectorizer = TfidfVectorizer(max_df=0.7, max_features=1300,
                             min_df=3, stop_words=mystopwords,
                             use_idf=True)
    
def pre_process_2(text):
    tokens = nltk.word_tokenize(text)
    tokens=[ WNlemma.lemmatize(t.lower()) for t in tokens]
    tokens=[ t for t in tokens if t not in mystopwords]
    tokens = [ t for t in tokens if len(t) >= 3 ]
    text_after_process=" ".join(tokens)
    return(text_after_process)

text = osha['Title']
toks = text.apply(pre_process_2)

X = vectorizer.fit_transform(toks)
X.shape

####################################### Apply KMeans for clustering
from sklearn.cluster import KMeans
from sklearn import metrics

#‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.
#Maximum number of iterations of the k-means algorithm for a single run.

km = KMeans(n_clusters=4, init='k-means++', max_iter=2000)
km.fit(X)

# Evaluate the clusters 
# Coefficient: more similar within clusters, more distant between clusters
# The higher the better (-1 to 1)

print("Coefficient for clusters: %0.3f"
      % metrics.silhouette_score(X, km.labels_))

####################################### How many docs in each cluster
labels, counts = np.unique(km.labels_[km.labels_>=0], return_counts=True)
print (labels)
print (counts)

######################################### What are the clusters about
# note: Clustering only gives you index of cluster rather than the meaning of cluster
# need to review the docs in each cluster and summarize 
# We still need to see the more representative words for each cluster to understand them.

def print_terms(cm, num):
    original_space_centroids = cm.cluster_centers_
    order_centroids = original_space_centroids.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(num):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :8]:
            print(' %s' % terms[ind], end='')
        print()

print_terms(km, 4)

dict = {0: 'finger, crushed, amputated, machine', 
        1: 'electric shock, burn',
        2: 'fall, ladder, fracture, roof, scaffold', 
        3: 'falling, tree, vehicle, truck, crane'
        }
        
print(dict)
print(counts)

####################################

from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag

ss = wn.synsets('employee')[0]
print(ss)
ss.hypernyms() # hypernym is the parent

ss.hyponyms() # hyponym is the child 

hyps = list(set(
                [w for s in ss.closure(lambda s:s.hyponyms())
                        for w in s.lemma_names()]))
# all hyponyms' lemma.names(); set() drop duplicates 
hyps 
len(hyps)


text2 = osha['Description']
text2

type(text2)

sent_pos = list()

for sent in text2: 
    sent_pos.append(pos_tag(word_tokenize(sent)))

sent_pos[:1]

wnl = nltk.WordNetLemmatizer()

def lemmaNVAR(wpos):
    final= []
    for wpo in wpos: 
        lemmas = []
        for w, pos in wpo:
            if pos[0] == 'N':
                lemmas.append(wnl.lemmatize(w.lower()))
        final.append(lemmas)     
    return final

n_pos = lemmaNVAR(sent_pos)
n_pos[:2]

from nltk import FreqDist
import string

stop = stopwords.words('english')+['would', 'could', 'hand']

def pre_process_3(toks):
    toks = [ t.lower() for t in toks if t not in string.punctuation+"’“”" ]
    toks = [t for t in toks if t not in stop ]
    toks = [ t for t in toks if not t.isnumeric() ]
    toks = [t for t in toks if t in hyps]
    return toks

n_pos_clean = [ pre_process_3(f) for f in n_pos ]
n_pos_clean

n_pos_unique = list()

for i in range(len(n_pos_clean)):
    n_pos_unique.append(set(n_pos_clean[i]))

n_pos_unique

n_pos_flat = [ c for l in n_pos_unique for c in l ]
n_pos_flat
n = FreqDist(n_pos_flat)
n.most_common(10)


