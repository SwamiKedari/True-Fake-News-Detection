#!/usr/bin/env python
# coding: utf-8

# # Installing the libraries 

# In[1]:


pip install nltk


# In[1]:


import nltk


# In[2]:


nltk.download()


# In[3]:


import pandas as pd


# In[4]:


fake=pd.read_csv("C:/Users/DELL/Downloads/fake.csv")


# In[5]:


true=pd.read_csv("C:/Users/DELL/Downloads/true.csv")


# In[6]:


fake.info()


# In[7]:


true.info()


# In[8]:


fake.head()


# In[9]:


true.head()


# In[10]:


fake.head(10)


# In[11]:


true.head(10)


# In[12]:


fake.subject.value_counts()


# # Adding new target column to both fake and true datasets

# In[13]:


fake["target"]=0
true["target"]=1


# In[14]:


fake.head(10)


# In[15]:


true.head(10)


# # Concatting the fake and the true datasets to make a single datasets

# In[16]:


data=pd.concat((fake,true),axis=0)


# In[17]:


data


# # Matching the indexes with the total number of rows in dataset

# In[18]:


data=data.reset_index(drop=True)


# In[19]:


data


# # Dropping the columns that are not needed in the training procedure

# In[20]:


data=data.drop(["subject","date","title"],axis=1)


# In[21]:


data


# In[22]:


data.columns


# # Applying the tokenizing procedure

# In[23]:


from nltk.tokenize import word_tokenize


# In[24]:


data["text"]=data["text"].apply(word_tokenize)


# In[25]:


data.head(10)


# In[26]:


from nltk.stem.snowball import SnowballStemmer
porter=SnowballStemmer("english")


# In[27]:


def stem_it(text):
    return [porter.stem(word) for word in text]


# In[28]:


data["text"]=data["text"].apply(stem_it)


# In[29]:


data.head()


# In[30]:


print(data.head(10))


# In[31]:


#stopword removal


# In[32]:


#from nltk.corpus import stopwords


# # Removing the stopwords

# In[33]:


def stop_it(t):
    dt=[word for word in t if len(word)>2]
    return dt


# In[34]:


data["text"]=data["text"].apply(stop_it)


# In[35]:


print(data.head(10))


# # Joining the list words to form the sentences

# In[36]:


data["text"]=data["text"].apply(" ".join)


# In[37]:


data.head(10)


# # Splitting of the data

# In[38]:


from sklearn.model_selection import train_test_split
    


# In[39]:


x_train,x_test,y_train,y_test=train_test_split(data["text"],data["target"],test_size=0.25)


# In[40]:


x_train.head()


# In[41]:


x_train.shape


# In[42]:


y_train.shape


# In[43]:


y_train.head()


# In[44]:


x_test.head()


# In[45]:


y_test.head()


# In[46]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[47]:


my_tfidf=TfidfVectorizer(max_df=0.7)


# In[48]:


tfidf_train=my_tfidf.fit_transform(x_train)
tfidf_test=my_tfidf.transform(x_test)


# In[49]:


print(tfidf_train)


# # Importing the libraries for training the model

# In[50]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# # Fitting and predicting the model accuracy

# In[53]:


model_1=LogisticRegression(max_iter=900)
model_1.fit(tfidf_train,y_train)
pred_1=model_1.predict(tfidf_test)
cr1=accuracy_score(y_test,pred_1)
print(cr1*100)


# In[54]:


from sklearn.linear_model import PassiveAggressiveClassifier
model=PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train,y_train)


# In[55]:


y_predict=model.predict(tfidf_test)
accscore=accuracy_score(y_test,y_predict)
print("the accuracy of the prediction is"+str(accscore*100))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




