#!/usr/bin/env python
# coding: utf-8

# In[1]:


min_complaint_length = 8
numTopics = 200
summary_length = 2


# In[2]:


import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm

import nltk
from nltk import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from textblob import TextBlob

import ast

from gensim import corpora,models
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

  
lemmatizer = WordNetLemmatizer()


# In[3]:


contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}


# In[4]:


def clean_text(text, remove_stopwords = True):    
    
    text = text.lower()
    
    
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)
    
    
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text = re.sub(r'xxxx', ' ', text)
    text = re.sub(r'xx', ' ', text)
    text = re.sub(r'xxx', ' ', text)
    
   
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    return text


# In[5]:


df = pd.read_csv('complaints.csv')
df = df[['Issue','Consumer complaint narrative']]
df = df[df['Consumer complaint narrative'].notna()]
df = df.reset_index(drop=True)


# In[6]:


df['tokenized_sent'] = df['Consumer complaint narrative'].apply(lambda x: sent_tokenize(x))
df['complaint_length'] = df['tokenized_sent'].apply(lambda x: len(x))
df = df[df['complaint_length']>=min_complaint_length]
df = df.reset_index(drop=True)
df.head()


# In[7]:


df1 = df.head(5000)
df1 = df1.drop(['complaint_length'],axis=1)
df1.head()


# In[8]:


fraud_scam = ['Improper use of your report','Fraud or scam','Problem with fraud alerts or security freezes','Threatened to contact someone or share information improperly','Taking/threatening an illegal action']
company_related = ['Took or threatened to take negative or legal action','Other features, terms, or problems','Confusing or misleading advertising or marketing','Problem with additional add-on products or services','Advertising','Identity theft protection or other monitoring services','Other service problem','Problem with customer service','Confusing or missing disclosures',"Problem with a company's investigation into an existing issue",'Advertising and marketing, including promotional offers','Communication tactics','Dealing with your lender or servicer','Problem with a lender or other company charging your account',"Can't contact lender or servicer"]
account = ['Closing your account',"Can't stop withdrawals from your bank account",'Problem with overdraft','Lost or stolen check','Problem adding money','Money was taken from your bank account on the wrong day or for the wrong amount','Lost or stolen money order','Closing an account','Opening an account','Managing an account','Incorrect information on your report','False statements or representation','Managing, opening, or closing your mobile wallet account']
transactions = ['Problem caused by your funds being low','Problem with a purchase shown on your statement','Trouble during payment process','Unexpected or other fees','Other transaction problem','Fees or interest','Problem when making payments',"Charged fees or interest you didn't expect",'Struggling to pay your bill','Unauthorized transactions or other transaction problem','Wrong amount charged or received','Problems when you are unable to pay',"Charged fees or interest I didn't expect",'Settlement process and costs','Problem with a purchase or transfer','Money was not available when promised']
debt = ['Written notification about debt','Attempts to collect debt not owed',"Cont'd attempts collect debt not owed",'Disclosure verification of debt']
loan = ['Vehicle was repossessed or sold the vehicle','Struggling to pay your loan','Problems at the end of the loan or lease','Managing the loan or lease','Shopping for a loan or lease', 'Struggling to repay your loan','Getting a loan or lease','Getting the loan',"Was approved for a loan, but didn't receive money",'Problem with the payoff process at the end of the loan',"Received a loan I didn't apply for",'Loan servicing, payments, escrow account','Taking out the loan or lease',"Loan payment wasn't credited to your account",'Getting a loan',"Received a loan you didn't apply for","Was approved for a loan, but didn't receive the money"]
credit_service_related = ["Problem with a credit reporting company's investigation into an existing problem",'Credit monitoring or identity theft protection services','Unable to get your credit report or credit score','Getting a line of credit']
card_related = ['Trouble using the card','Problem getting a card or closing an account','Getting a credit card','Trouble using your card']
mortgage = ['Closing on a mortgage','Struggling to pay mortgage','Applying for a mortgage or refinancing an existing mortgage','Applying for a mortgage','Application, originator, mortgage broker','Payoff process']
other = ['Other']


# In[9]:


labels_list = [loan,credit_service_related,card_related,mortgage,debt,transactions,account,company_related,fraud_scam,other]
label = ['loan','credit_service','card','mortgage','debt','transactions','account','company_issue','fraud_scam','others']


# In[10]:


def assign_label(x,label_list,label):
    for i in range(len(labels_list)):
        for j in labels_list[i]:
            if x==j:
                return label[i]
        


# In[11]:


df1.insert(1, "label", [None]*len(df1), True)


# In[12]:


df1['label'] = df1['Issue'].apply(lambda x: assign_label(x,labels_list,label))


# In[13]:


df1 = df1.drop(['Issue'],axis=1)
df1.head()


# In[14]:


df1.rename(columns = {'label':'Issue'}, inplace = True)


# In[16]:


for i in range(len(df1['tokenized_sent'])):
    comp = df1['tokenized_sent'][i]
    for j in comp:
        v = word_tokenize(j)
        if len(v)==1:
            df1['tokenized_sent'][i].remove(j)


# In[32]:


def extract_ngrams(data, num):
    n_grams = TextBlob(data).ngrams(num)
    return [ ' '.join(grams) for grams in n_grams]


# In[ ]:


f =  open('keywords_file.txt')
keyword = f.readlines()    
f.close()

keywords_list = [None]*len(keyword)
for i in range(len(keyword)):
    keywords_list[i] = ast.literal_eval(keyword[i])


# In[ ]:


remove = []
for i in range(len(df1)):
    comp = df1['Consumer complaint narrative'][i]
    issue_label = df1['Issue'][i]    
    comp = clean_text(comp)
    comp = lemmatizer.lemmatize(comp)
    comp2 = extract_ngrams(comp,2)
    comp1 = extract_ngrams(comp,1)
    comp = comp1+comp2
    
    for j in range(len(label)):
        if issue_label==label[j]:
            key = keywords_list[j]
            match = list(set(comp).intersection(key))
            
            if len(match) == 0:
                remove.append(i)
            


# In[ ]:


indexes_to_keep = set(range(df1.shape[0])) - set(remove)
df1 = df1.take(list(indexes_to_keep))
df1.reset_index(inplace = True, drop = True) 


# In[ ]:


df1.insert(3, "words_of_sents", [None]*len(df1), True)
df1.insert(4, "vectors_of_corpus", [None]*len(df1), True)
df1.insert(5, "vectors_of_sents", [None]*len(df1), True)
df1.insert(6, "summary", [None]*len(df1), True)


# In[17]:


def tokenization(sentences_list):
    
    words_list = [None]*len(sentences_list)
    for i in range(len(sentences_list)):
        words_list[i] = word_tokenize(sentences_list[i])
    
    return words_list


# In[18]:


def takeFirst(x):
    return x[0]

def takeSecond(x):
    return x[1]

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

def lsi(sentTokens,numTopics):
    
    dct = corpora.Dictionary(sentTokens)
    corpus = list(map(lambda st: dct.doc2bow(st), sentTokens))    
    lsi = models.LsiModel(corpus, id2word=dct,num_topics=numTopics)
    
    vecCorpus = lsi[corpus]
   
    return vecCorpus         


# In[19]:


def Cumulative(lists):  
    cu_list = []  
    length = len(lists)  
    cu_list = [sum(lists[0:x:1]) for x in range(0, length+1)]  
    return cu_list[1:] 


# In[20]:


def vectors_of_sentences(x,numTopics):
    vectors = [None]*len(x)
    for i,dv in enumerate(x):
        array = [None]*numTopics
        for sc in dv:
            array[sc[0]] = sc[1]
        vectors[i] = array
        
    return vectors  


# In[21]:


def get_complaint_vectors(tokenized_sent,numTopics):
    
    df1['words_of_sents'] = tokenized_sent.apply(lambda y: tokenization(y))
    complaint_corpus = []
    
    for i in range(len(df1['words_of_sents'])):
        complaint_corpus += df1['words_of_sents'][i]

    vectors_of_complaints = lsi(complaint_corpus,numTopics)

    lengths_of_sentences = []
    
    for i in range(len(df1['words_of_sents'])):
        lengths_of_sentences.append(len(df1['words_of_sents'][i]))
    lengths_of_sentences.insert(0,0)

    l = Cumulative(lengths_of_sentences)


    for i in range(len(lengths_of_sentences)-1):
        df1['vectors_of_corpus'][i] = vectors_of_complaints[l[i]:l[i+1]]
            
    df1['vectors_of_sents'] = df1['vectors_of_corpus'].apply(lambda x: vectors_of_sentences(x,numTopics))


# In[22]:


def lsi_summ(vecCorpus,numTopics,summary_length,sents):    

    b=[]
    
    for i in range(len(df1['tokenized_sent'])):
        if len(df1['tokenized_sent'][i])==1:
            b.append(i)
    
    l = list(range(len(df1)))
    
    r = list(set(l)^set(b))
    
    for i in r:
        
        sortedVecs = list(map(lambda x: list(), range(numTopics)))

        for j,dv in enumerate(vecCorpus[i]):
            for sc in dv:
                isc = (j, abs(sc[1]))
                sortedVecs[sc[0]].append(isc)

        sortedVecs = list(map(lambda iscl: sorted(iscl,key=takeSecond,reverse=True), sortedVecs))
          
    
        top_sents = selTopSents(summary_length,numTopics,sortedVecs)       
        top_sents = sorted(top_sents,key=takeFirst)
        top_sentences = list(map(lambda ts: (ts[0],sents[i][ts[0]], ts[1]), top_sents))
        
        df1['summary'][i] = top_sentences
        
   


# In[23]:


summ_df = pd.DataFrame()
summ_df.insert(0, "summary", [None]*(len(df1)), True)


# In[24]:


def summ_modification_index():
    
    x = []
    modified = 0
    for i in range(len(df1)):
        temp1 = summ_df['summary'][i]
        temp2 = df1['summary'][i]
        x1 = []
        x2 = []
        for j in range(summary_length):
            x1.append(temp1[j][1])
            x2.append(temp2[j][1])
        
        x.append(list(set(x1)^set(x2)))
        
    for i in range(len(df1)):
        if x[i]!=[]:
            modified+=1
    return modified/len(df1)        


# In[25]:


class garbage_dictionary(dict): 
  
    
    def __init__(self): 
        self = dict() 
          
    
    def add(self, key, value): 
        self[key] = value 
        
blacklist = garbage_dictionary()
for i in range(len(df1)):
        blacklist.add(i,[None])


# In[26]:


def update_blacklist(bl):
    
    f =  open('blacklist.txt')
    black = f.readlines()    
    f.close()
    
    i = input("pruning round # : ")
    b = black[int(i)]
    b = b.split(' ')
    
    for j in b:
        if j=='\n':
            b.remove(j)

    for j in b:
        j = j.split(':')
        key = int(j[0])
        value = ast.literal_eval(j[1])
        bl.add(key,value)


# In[27]:


def sent_remove(comp,lis):
    sent_to_remove = []
    if lis!=[]:
        for j in lis:
            sent_to_remove.append(comp[j])
        for k in sent_to_remove:
            comp.remove(k)
            
def vect_remove(comp,lis):
    vect_to_remove = []
    if lis!=[]:
        for j in lis:
            vect_to_remove.append(comp[j])
        for k in vect_to_remove:
            comp.remove(k)


# In[28]:


def remove_sents(blacklist,vec,numTopics):
    
    b=[]
    for k in blacklist.keys():
        if blacklist[k]!=[None]:
            b.append(k)
    
    l = list(range(len(df1)))
    
    r = list(set(l)^set(b))
    
    cosine_sim = list(map(lambda i: list(), range(len(df1))))
    
    for i in r:
        vectors = df1['vectors_of_sents'][i]        
        for j in range(len(vectors)):
            vec1 = vectors[j]
            sim = dot(vec1,vec)/(norm(vec1)*norm(vec))           
            cosine_sim[i].append([i,sim,j])
    
    x = []
    for i in r:
        x += comp_sim[i]
    
    top_sim = sorted(x,key=takeSecond,reverse=True)
    x = None
    top_sim = top_sim[0:10]
    
    cos_sim = list(map(lambda i: list(), range(len(df1))))
    
    for i in range(len(top_sim)):
        index = top_sim[i][0]
        sentence_id = top_sim[i][1]
        cos_sim[index].append(sentence_id)           
    
    for i in range(len(cos_sim)):
        
        sent_remove(df1['tokenized_sent'][i],cos_sim[i])
        vect_remove(df1['vectors_of_sents'][i],cos_sim[i])      


# In[29]:


def prune_data(blacklist,numTopics,similarity_threshold):
    
    for i in range(len(blacklist)):
        
        if blacklist[i]!=[None]:
            
            sent_remove(df1['tokenized_sent'][i],blacklist[i]) 
            v = df1['vectors_of_sents'][i]
            
            for j in blacklist[i]:                                             
                ref_v = v[j]
                remove_sents(blacklist,ref_v,numTopics)
                
            vect_remove(df1['vectors_of_sents'][i],blacklist[i])
                
    
    for i in range(len(df1)):
        summ_df['summary'][i] = df1['summary'][i]
    
    blacklist.clear()
    for i in range(len(df1)):
        blacklist.add(i,[None])


# In[34]:


modification_index = []
relevance = []


# In[35]:


def get_relevance_score():
    
    correct_pred = 0
    for j in range(len(df1)):
            x = df1['summary'][j]
            issue = df1['Issue'][j]

            matches = [None]*len(keywords)
            summ =' '
            for i in range(len(x)):
                summ+= x[i][1]
            summ = clean_text(summ)
            summ = lemmatizer.lemmatize(summ)
            summ1 = extract_ngrams(summ,1)
            summ2 = extract_ngrams(summ,2)
            summ = summ1+summ2
            
            for i in range(len(keywords)):
                matches[i] = list(set(summ).intersection(keywords[i]))

            no_matches = []

            for i in range(len(matches)):
                no_matches.append(len(matches[i]))
               
            m = max(no_matches)
            
            if m!=0:
                match_list = []
                for k in range(len(no_matches)):
                    if no_matches[k] == m:
                        match_list.append(labels[k])

                for k in match_list:
                    if k==issue:
                        correct_pred+=1
                        break          
    
    
    return correct_pred/len(df1)   
    


# In[114]:


get_complaint_vectors(df1['tokenized_sent'],numTopics)


# In[115]:


lsi_summ(df1['vectors_of_corpus'],numTopics,summary_length,df1['tokenized_sent'])


# In[116]:


modification_index.append(summ_modification_index())


# In[118]:


relevance.append(get_relevance_score())


# In[ ]:


add_blacklist = open("blacklist.txt",mode="a",encoding="utf-8")
add_blacklist.write('\n')
add_blacklist.close()


# In[ ]:


# update_blacklist(blacklist)


# In[113]:


prune_data(blacklist,numTopics,similarity_threshold)


# In[ ]:


"""
Function for extracting keywords from complaints for all the labels
Groups complaints belonging to a particular label and prepares a TFIDF table
Then returns the top 'n' features where n is defined by the user

"""

# def assign_keywords(x,y,keywords_n_grams):
    
#     vectorizer = TfidfVectorizer(ngram_range =(keywords_n_grams,keywords_n_grams))
#     vectors = vectorizer.fit_transform([x])
#     feature_names = vectorizer.get_feature_names()
#     dense = vectors.todense()
#     denselist = dense.tolist()


#     df2 = pd.DataFrame(denselist, columns=feature_names)
#     df2 = df2.sort_values(by = 0 ,axis=1, ascending=False, inplace=False, kind='quicksort', na_position='last')

#     features =[]
#     for col in df2.columns:
#         features.append(col)
        
#     return features[0:y]    



# df1['cleaned_complaints'] = df1['Consumer complaint narrative'].apply(lambda x: clean_text(x))

# labels= []
# for i in df1['Issue'].unique():
#     labels.append(i)
    
# issue_comp = list(map(lambda i: ' ', range(len(labels))))

# for i in range(len(df1)):
#     l = df1['Issue'][i]
#     for j in range(len(labels)):
#         lab = labels[j]
#         if l==lab:
#             issue_comp[j]+= ' ' + df1['cleaned_complaints'][i]     
            

# keywords = list(map(lambda i: ' ', range(len(labels))))

# for i in range(len(labels)):
#     keywords[i] = assign_keywords(lemmatizer.lemmatize(issue_comp[i]),30,1)        

