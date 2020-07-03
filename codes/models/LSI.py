#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk
from nltk import sent_tokenize,word_tokenize
from gensim import corpora,models


# In[2]:


df = pd.read_csv('complaints.csv')
df.head()


# In[3]:


df = df[['Issue','Consumer complaint narrative']]
df = df[df['Consumer complaint narrative'].notna()]
df = df.reset_index(drop=True)


# In[4]:


min_complaint_length = 8
summary_length = 2
numTopics = 5


# In[5]:


df['tokenized_sent'] = df['Consumer complaint narrative'].apply(lambda x: sent_tokenize(x))
df['complaint_length'] = df['tokenized_sent'].apply(lambda x: len(x))
df = df[df['complaint_length']>= min_complaint_length]
df = df.reset_index(drop=True)
df.head()


# In[6]:


df1 = df.head(1000)


# In[7]:


def tokenization(sentences_list):
    
    words_list = [None]*len(sentences_list)
    for i in range(len(sentences_list)):
        words_list[i] = word_tokenize(sentences_list[i])
    
    return words_list


# In[8]:


df1['words_of_sents'] = df1['tokenized_sent'].apply(lambda y: tokenization(y))


# In[9]:


df1['zipped_tokens'] = list(zip(df1.tokenized_sent, df1.words_of_sents))
df1.head()


# In[10]:


def takeFirst(x):
    return x[0]

def takeSecond(x):
    return x[1]


# In[11]:


def selTopSents(summSize, numTopics, sortedVecs):
    topSentences = []
    sentIndexes = set()
    sCount = 0
    for i in range(summSize):
        for j in range(numTopics):
            vecs = sortedVecs[j]
            si = vecs[i][0]
            if si not in sentIndexes:
                topSentences.append(vecs[i])
                sentIndexes.add(si)
                sCount += 1
            if sCount == summSize:
                return topSentences


# In[29]:


def lsi_summ(sentTokens,numTopics,sents):
    
    dct = corpora.Dictionary(sentTokens)
    corpus = list(map(lambda st: dct.doc2bow(st), sentTokens))    
    lsi = models.LsiModel(corpus, id2word=dct,num_topics=numTopics)
    
    vecCorpus = lsi[corpus]
    
    sortedVecs = list(map(lambda i: list(), range(numTopics)))
    for i,dv in enumerate(vecCorpus):
        for sc in dv:
            isc = (i, abs(sc[1]))
            sortedVecs[sc[0]].append(isc)
    sortedVecs = list(map(lambda iscl: sorted(iscl,key=takeSecond,reverse=True), sortedVecs))
    
    top_sents = selTopSents(summary_length,numTopics,sortedVecs)
    top_sents = sorted(top_sents,key=takeFirst)
    top_sentences = list(map(lambda ts: (sents[ts[0]], ts[1]), top_sents)) 
    
   
    return top_sentences
    


# In[13]:


df1['lsi_model'] = df1['zipped_tokens'].apply(lambda x: lsi_summ(x[1],numTopics,x[0]))
df1.head()


# In[14]:


for i in range(5):
    print(df1['lsi_model'][i])


# In[15]:


from gensim.parsing.preprocessing import preprocess_string, strip_punctuation,strip_numeric

lda_topics = df1['lsi_model'][3].show_topics(num_words=5)

topics = []
filters = [lambda x: x.lower(), strip_punctuation, strip_numeric]

for topic in lda_topics:
    print(topic)
    topics.append(preprocess_string(topic[1], filters))

print(topics)


# In[17]:


df1['vectors_of_corpus'] = df1['zipped_tokens'].apply(lambda x: lsi_summ(x[1],numTopics,x[0]))
df1.head()


# In[18]:


for i,dv in enumerate(df1['vectors_of_corpus'][3]):
    print(i)
    print(dv)


# In[19]:


def vectors_of_sentences(x,numTopics):
    vectors = [None]*len(x)
    for i,dv in enumerate(x):
        array = [None]*numTopics
        for sc in dv:
            array[sc[0]] = sc[1]
        vectors[i] = array
        
    return vectors    
    
    


# In[20]:


df1['sentence_vectors'] = df1['vectors_of_corpus'].apply(lambda x: vectors_of_sentences(x,numTopics))
df1.head()


# In[22]:


for i in (df1['sentence_vectors'][3]):
    print(i)


# In[24]:


df1['sorted_sentence_vectors'] = df1['zipped_tokens'].apply(lambda x: lsi_summ(x[1],numTopics,x[0]))
df1.head()


# In[25]:


for i in (df1['sorted_sentence_vectors'][3]):
    print(i)


# In[27]:


df1['top_sorted_sentence_vectors'] = df1['zipped_tokens'].apply(lambda x: lsi_summ(x[1],numTopics,x[0]))
df1.head()


# In[28]:


df1['top_sorted_sentence_vectors'][3]


# In[30]:


df1['summary'] = df1['zipped_tokens'].apply(lambda x: lsi_summ(x[1],numTopics,x[0]))
df1.head()


# In[31]:


df1['Consumer complaint narrative'][2]


# In[33]:


df1['summary'][3]


# In[ ]:




