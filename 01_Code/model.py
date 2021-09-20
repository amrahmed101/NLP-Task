#!/usr/bin/env python
# coding: utf-8

# <center> <h1> <br>Machine Learning Internship Task <br>
# <br>Job title : Classification by industry<br>
# <br>(Multi-text Text Classification Task)<br> </h1> </center>

# <br>Task Description:<br>
# <br>Leveraging the use of a machine learning classifier to classify job titles by the industry.<br>

# <br>Importing Libraries<br>

# In[39]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[38]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.pipeline import make_pipeline


# In[ ]:


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler 
import xgboost as xgb
from sklearn.svm import SVC


# In[49]:


from sklearn.metrics import accuracy_score


# In[40]:


import gensim
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import nltk
import re


# <center> <h1> Exploratory Data Analysis </h1> </center> 
# 
# 
# 
# ### Data Attributes:
# 
#     1.  Job Title: the input predictor for the task.
#     2.  Industry: the corresponding industry for the job title which will be the target class in this case.

# In[1]:


#importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from matplotlib.pyplot import figure
import warnings
warnings.filterwarnings("ignore")


# In[2]:


#Reading the data, viewing the top part of the data
data  = pd.read_csv(r"Job titles and industries.csv")
data.head()


# In[3]:


#data information
data.info()


# In[4]:


#data columns
data_heads=data.columns
data_heads


# In[5]:


#checking if the data has null values
data[data_heads].isnull().sum()


# In[6]:


#Data columns count, unique values and frequency of most occuring records 
data[data_heads].describe().T


# In[7]:


data[data_heads].value_counts()[:30]


# In[8]:


#plotting the most frequent titles by their probability
prob = data[data_heads].value_counts(normalize=True)[:30]
plt.figure(figsize=(8,6))
prob.plot(kind='bar')
plt.xticks(rotation=90)
plt.show()


# In[9]:


#highest occuring comment


# In[10]:


#counting the labels
sns.countplot(data["industry"])
plt.show()


# In[ ]:


#imbalanced comment


# <center> <h1>Preprocessing </h1> </center>
# 
# <br>Preprocessing will consist of two main parts<br>
# <br>-Text preprocessing<br>
# <br>-Data preprocessing<br>

# ## Text Preprocessing

# ## Removing Stop Words
# <br>-Stop words are common words such as a, and, and the which have little significance to our job titles<br>
# <br>-By combining the simple_preprocess function from gensim library and the stop_words set from the NLTK library to filter stopwords out of each job title.

# In[14]:


stop_words = set(stopwords.words('english'))


# In[31]:


def text_cleaning(frase):
    result = ""
    for token in simple_preprocess(frase):
        if token not in stop_words and len(token) >= 2:
            token = token.lower() 
            result += (token + " ")    
    return result


# In[32]:


data["job title"] = data["job title"].map(text_cleaning)


# ## Data preprocessing

# ## Dealing with Class imbalance
# 
# As noticed before in the EDA there is a signifcant class imbalance in the target labels.
# <br>This will be rectified using the SMOTE library which is designed specifically to deal with Imbalanced Data.
# <br>The SMOTE function will be added to the pipeline.

# In[43]:


#Using Test-Train split for splitting the data

X_train, X_test, y_train, y_test = train_test_split(data['job title'],data['industry'], test_size=0.2)


# <center> <h1>Pipeline Steps </h1> </center>
# <br>1) TfidfVectorizer() : Convert a collection of raw documents to a matrix of TF-IDF features.
# <br>In order to use this data for machine learning, we need to be able to convert the content of each string into a vector of numbers. 
# <br>For this we will use the TF-IDF vectorizerand.<br>
# <br>2) SMOTE(): Synthetic Minority Oversampling Technique
# <br> solve the problem of imbalanced data by oversampling the examples in the minority class
# 
# <br>3) Machine Learning Models :
# <br>Using three different machine learning models for training
# <br>-MultinomialNB: multinomial Naive Bayes classifier.
# <br>-RandomForestClassifier: a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset.
# <br>-SVC: C-Support Vector Classification.

# In[81]:


#Creating Three Models for each machine learning model

model_mb = make_pipeline(TfidfVectorizer(),SMOTE(random_state=2), MultinomialNB())
model_svc = make_pipeline(TfidfVectorizer(),SMOTE(random_state=2),  SVC())
model_rf = make_pipeline(TfidfVectorizer(),SMOTE(random_state=2),  RandomForestClassifier())


# In[82]:


#Fitting the models

model_mb.fit(X_train, y_train)
model_svc.fit(X_train, y_train)
model_rf.fit(X_train, y_train)


# In[83]:


#predicting labels

labels_mb = model_mb.predict(X_test)
labels_svc = model_svc.predict(X_test)
labels_rf = model_rf.predict(X_test)


# In[96]:


#Evaluating Model Accuracy

from sklearn.metrics import classification_report
print("Multinomial Naive Bayes Classification Report\n")
print(classification_report(y_test, labels_mb))
print("\nSupport Vector Classifier Classification Report\n")
print(classification_report(y_test, labels_svc ))
print("\nRandom Forest Classifier Classification Report\n")
print(classification_report(y_test, labels_rf))


# In[ ]:


##Conclusion Support Vector Classifier


# In[98]:


import pickle
pickle.dump(model_svc, open('model.pkl','wb'))


# In[ ]:




