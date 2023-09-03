#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


books=pd.read_csv('BX-Books.csv',sep=';' ,encoding='latin-1',on_bad_lines='skip')


# In[3]:


books.head()


# In[4]:


books=books[['ISBN','Book-Title','Book-Author','Year-Of-Publication','Publisher']]


# In[5]:


books


# In[6]:


books.head(2)


# In[7]:


books.rename(columns={'Book-Title':'title','Book-Author':'author','Year-Of-Publication':'year','Publisher':'publisher'},inplace=True)


# In[8]:


books


# In[9]:


users=pd.read_csv('BX-Users.csv',sep=';' ,encoding='latin-1',on_bad_lines='skip')


# In[10]:


users.head()


# In[11]:


users.columns


# In[12]:


users.rename(columns={'User-ID':'id','Location':'location','Age':'age'},inplace=True)


# In[13]:


users.head(2)


# In[14]:


ratings=pd.read_csv('BX-Book-Ratings.csv',sep=';' ,encoding='latin-1',on_bad_lines='skip')


# In[15]:


ratings.head()


# In[16]:


ratings.rename(columns={'User-ID':'id','Book-Rating':'rating'},inplace=True)


# In[17]:


ratings


# In[18]:


books.shape


# In[19]:


users.shape


# In[20]:


ratings.shape


# In[21]:


x=ratings['id'].value_counts()>200


# In[22]:


y=x[x].index


# In[23]:


ratings=ratings[ratings['id'].isin(y)]


# In[24]:


ratings.shape


# In[25]:


rating_with_books=ratings.merge(books,on='ISBN')


# In[26]:


rating_with_books.head()


# In[27]:


number_of_ratings=rating_with_books.groupby('title')['rating'].count().reset_index()


# number_of_ratings.head()

# In[28]:


number_of_ratings.rename(columns={'rating':'number of ratings'},inplace=True)


# In[29]:


number_of_ratings.head()


# In[30]:


final_rating=rating_with_books.merge(number_of_ratings,on='title')


# In[31]:


final_rating.shape


# In[32]:


final_rating=final_rating[final_rating['number of ratings']>=50]


# In[33]:


final_rating.shape


# In[34]:


final_rating.drop_duplicates(['id','title'],inplace=True)


# In[35]:


final_rating.shape


# In[36]:


book_pivot=final_rating.pivot_table(columns='id',index='title',values='rating')


# In[37]:


book_pivot.head()


# In[38]:


book_pivot.fillna(0,inplace=True)


# In[39]:


book_pivot.head()


# In[40]:


from scipy.sparse import csr_matrix
book_sparse=csr_matrix(book_pivot)


# In[41]:


type(book_sparse)


# In[49]:


from sklearn.neighbors import NearestNeighbors


# In[50]:


model=NearestNeighbors(algorithm='brute')


# In[53]:


model.fit(book_sparse)


# In[65]:


distances , suggestions=model.kneighbors(book_pivot.iloc[45,:].values.reshape(1,-1), n_neighbors=6)


# In[66]:


distances


# In[67]:


suggestions


# In[68]:


for i in range(len(suggestions)):
    print(book_pivot.index[suggestions[i]])


# In[69]:


book_pivot.index[45]


# In[99]:


def recommender_books(book_name):
    book_id=np.where(book_pivot.index==book_name)[0][0]
    distances , suggestions=model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6)
    for i in range(len(suggestions)):
        if i==0:
            print('The Suggested Books for {}:'.format(book_name))
        if not i:
            print(book_pivot.index[suggestions[i]])


# In[100]:


recommender_books('Animal Farm')


# In[ ]:




