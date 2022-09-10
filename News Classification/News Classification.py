#!/usr/bin/env python
# coding: utf-8

# In[76]:


pip install nltk


# In[77]:


import nltk


# In[78]:


nltk.download('punkt')


# In[79]:


import pandas as pd


# In[80]:


fake = pd.read_csv("Fake.csv")
genuine = pd.read_csv("True.csv")


# In[81]:


display(fake.info())


# In[82]:


display(genuine.info())


# In[83]:


display(genuine.head(10))


# In[84]:


display(fake.subject.value_counts())


# In[85]:


fake['target']=0
genuine['target']=1


# In[86]:


display(genuine.head(10))


# In[87]:


display(fake.head(10))


# In[88]:


data= pd.concat([fake,genuine],axis=0)


# In[89]:


date=data.reset_index(drop=True)


# In[90]:


data=data.drop(['subject','date','title'],axis=1)


# In[91]:


print(data.columns)


# In[92]:


from nltk.tokenize import word_tokenize


# In[93]:


data['text']=data['text'].apply(word_tokenize)


# In[94]:


print(data.head(10))


# In[95]:


from nltk.stem.snowball import SnowballStemmer
porter =SnowballStemmer("english")


# In[96]:


def stem_it(text):
    return [porter.stem(word) for word in text]


# In[97]:


data['text']=data['text'].apply(stem_it)


# In[98]:


print(data.head(10))


# In[99]:


def stop_it(t):
    dt=[word for word in t if len(word)>2]
    return dt


# In[100]:


data['text']=data['text'].apply(stop_it)


# In[101]:


print(data['text'].head(10))


# In[102]:


data['text']=data['text'].apply(' '.join)


# In[103]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(data['text'], data['target'], test_size=0.25)
display(X_train.head())
print('\n')
display(y_train.head())


# In[104]:


from sklearn.feature_extraction.text import TfidfVectorizer
my_tfidf = TfidfVectorizer(max_df=0.7)

tfidf_train = my_tfidf.fit_transform(X_train)
tfidf_test = my_tfidf.transform(X_test)


# In[105]:


print(tfidf_train)


# In[106]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[108]:


model_1 = LogisticRegression(max_iter=900)
model_1.fit(tfidf_train, y_train)
pred_1 = model_1.predict(tfidf_test)
cr1 = accuracy_score(y_test,pred_1)
print(cr1*100)


# In[109]:


from sklearn.linear_model import PassiveAggressiveClassifier

model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train, y_train)


# In[110]:


y_pred = model.predict(tfidf_test)
accscore = accuracy_score(y_test, y_pred)
print('The accuracy of prediction is ',accscore*100)


# In[ ]:




