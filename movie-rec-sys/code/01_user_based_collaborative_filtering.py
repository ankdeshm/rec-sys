#!/usr/bin/env python
# coding: utf-8

# ## User-Based Collaborative Filtering

# ### Import necessary modules

# In[10]:


#data analysis libraries 
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Enable multiple output cells
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[11]:


# Load training dataset which contains the ratings for movies by different users
training_full = pd.read_csv("/Users/ankitadeshmukh/Desktop/SJSU/Academic/Fall22/CMPE257/Project/Dataset/serendipity-sac2018/training.csv")
training_full.head()


# In[12]:


# Drop unnecessary columns
cols_to_drop = ['timestamp']
training_full.drop(cols_to_drop, axis = 1, inplace = True)
training_full.head()


# In[13]:


training_full.shape


# In[14]:


n_users = training_full['userId'].nunique()
n_movies = training_full['movieId'].nunique()

print('Number of users:', n_users)
print('Number of movies:', n_movies)


# In[15]:


train_sample_df = training_full.iloc[:1000000] 
train_sample_df.shape


# In[16]:


from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(train_sample_df, test_size = 0.30, random_state = 42)

print(X_train.shape)
print(X_test.shape)


# In[17]:


# pivot ratings into movie features
user_data = X_train.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(0)
user_data.head()


# In[18]:


# make a copy of train and test datasets
dummy_train = X_train.copy()
dummy_test = X_test.copy()


# In[19]:


dummy_train['rating'] = dummy_train['rating'].apply(lambda x: 0 if x > 0 else 1)
dummy_test['rating'] = dummy_test['rating'].apply(lambda x: 1 if x > 0 else 0)
# The movies not rated by user is marked as 1 for prediction 
dummy_train = dummy_train.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(1)
# The movies not rated by user is marked as 0 for evaluation 
dummy_test = dummy_test.pivot(index ='userId', columns = 'movieId', values = 'rating').fillna(0)


# In[20]:


dummy_train.head()


# In[21]:


dummy_test.head()


# In[22]:


# User-User Similarity matrix using Cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

# User Similarity Matrix using Cosine similarity as a similarity measure between Users
user_similarity = cosine_similarity(user_data)
user_similarity[np.isnan(user_similarity)] = 0
print(user_similarity)
print(user_similarity.shape)


# In[23]:


# Predicting the User ratings on the movies
user_predicted_ratings = np.dot(user_similarity, user_data)
user_predicted_ratings


# In[24]:


user_predicted_ratings.shape


# In[25]:


# np.multiply for cell-by-cell multiplication 

user_final_ratings = np.multiply(user_predicted_ratings, dummy_train)
user_final_ratings.head()


# In[26]:


user_final_ratings.iloc[42].sort_values(ascending = False)[0:5]


# In[27]:


# Item-based collaborative filtering
movie_features = X_train.pivot(index = 'movieId', columns = 'userId', values = 'rating').fillna(0)
movie_features.head()


# In[28]:


# Item-Item Similarity matrix using Cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

# Item Similarity Matrix using Cosine similarity as a similarity measure between Items
item_similarity = cosine_similarity(movie_features)
item_similarity[np.isnan(item_similarity)] = 0
print(item_similarity)
print("- "*10)
print(item_similarity.shape)


# In[29]:


# Predicting the User ratings on the movies
item_predicted_ratings = np.dot(movie_features.T, item_similarity)
item_predicted_ratings


# In[30]:


item_predicted_ratings.shape


# In[31]:


dummy_train.shape


# In[32]:


# Filtering the ratings only for the movies not already rated by the user for recommendation
# np.multiply for cell-by-cell multiplication 

item_final_ratings = np.multiply(item_predicted_ratings, dummy_train)
item_final_ratings.head()


# In[33]:


# Top 5 movie recommendations for the User 42
item_final_ratings.iloc[42].sort_values(ascending = False)[0:5]


# ### Evaluation
# #### Evaluation will we same as you have seen above for the prediction. The only difference being, you will evaluate for the movie already rated by the User instead of predicting it for the movie not rated by the user.

# In[34]:


# Using User-User similarity
test_user_features = X_test.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(0)
test_user_similarity = cosine_similarity(test_user_features)
test_user_similarity[np.isnan(test_user_similarity)] = 0

print(test_user_similarity)
print("- "*10)
print(test_user_similarity.shape)


# In[35]:


user_predicted_ratings_test = np.dot(test_user_similarity, test_user_features)
user_predicted_ratings_test


# In[36]:


# Testing on the movies already rated by the user
test_user_final_rating = np.multiply(user_predicted_ratings_test, dummy_test)
test_user_final_rating.head()


# In[37]:


train_sample_df['rating'].describe()


# In[38]:


# But we need to normalize the final rating values between range (0.5, 5)

from sklearn.preprocessing import MinMaxScaler

X = test_user_final_rating.copy() 
X = X[X > 0] # only consider non-zero values as 0 means the user haven't rated the movies

scaler = MinMaxScaler(feature_range = (0.5, 5))
scaler.fit(X)
pred = scaler.transform(X)

print(pred)


# In[39]:


# total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(pred))
total_non_nan


# In[40]:


test = X_test.pivot(index = 'userId', columns = 'movieId', values = 'rating')
test.head()


# In[41]:


# RMSE Score

diff_sqr_matrix = (test - pred)**2
sum_of_squares_err = diff_sqr_matrix.sum().sum() # df.sum().sum() by default ignores null values

rmse = np.sqrt(sum_of_squares_err/total_non_nan)
print(rmse)


# In[42]:


# Mean abslute error

mae = np.abs(pred - test).sum().sum()/total_non_nan
print(mae)


# ### Conclusion
# #### It means that on an average our User-based recommendation engine is making an error of 1.2 in predicting the ratings by users.
