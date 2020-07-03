#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd


# In[55]:


df = pd.read_csv('test.csv')


# In[56]:


df = df.dropna()


# In[57]:


df


# In[58]:


df1 = pd.DataFrame() 
df1['complaint'] = df['complaint']


# In[59]:


df1


# In[60]:


df1['complaint'][1]


# In[66]:


def classify(complaint,business_words,qualifiers,threshold):
    
    length = len(complaint)
    customer = [i for i in range(length) if complaint.startswith('customer', i)] 
    agent = [i for i in range(length) if complaint.startswith('agent', i)]
    tags = customer+agent
    tags.append(length-1)
    tags.sort()
    
    conv = [None] * (len(tags)-1)
    
    for i in list(range(len(tags)-1)):
        conv[i] = complaint[tags[i]:tags[i+1]]
    
    
    for i in list(range(len(conv))):
        
        talk = conv[i]        
        b_first_occurence = [None] * (len(business_words))
        q_first_occurence = [None] * (len(qualifiers))
     
        b_first = 0
        q_first = 0
        context = 0
        indi = "context not found"
        b_query = "null"
        q_query = "null"
        
        k = 0
        
        for j in business_words:
            b_first_occurence[k] = talk.find(j)
            k = k+1

        k = 0
        for j in qualifiers:
            q_first_occurence[k] = talk.find(j)
            k = k+1

        b_first_occurence = [item for item in b_first_occurence if item >= 0]
        q_first_occurence = [item for item in q_first_occurence if item >= 0]

        if b_first_occurence == [] or q_first_occurence == []:
            continue       

        else: 

            b_first = min(b_first_occurence)
            q_first = min(q_first_occurence)
            
            if abs(b_first-q_first)<= threshold:
                context = 1
                b_query = ""
                q_query = ""
                k = b_first
                while talk[k]!= " ":
                    b_query = b_query + talk[k]
                    k+=1
                k = q_first
                while talk[k] != " ":
                    q_query = q_query + talk[k]
                    k+=1
                if talk[0]=='a':
                    indi = 'context came from agent'
                else:
                    indi = 'context came from customer'
                break
            else:
                continue
    
    return context,indi,b_query,q_query 
      


# In[72]:


business_words = ['credit' , 'limit','credit limit','mortgage','interest rate']
qualifiers = ['reduce','decreased','lower','higher','increased','increment','reduction','increment']
threshold = 40
df1['complaint'].apply(lambda x: classify(x,business_words,qualifiers,threshold))


# In[ ]:





# In[ ]:




