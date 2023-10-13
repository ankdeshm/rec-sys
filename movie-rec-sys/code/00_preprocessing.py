#!/usr/bin/env python
# coding: utf-8

# ### Import necessary modules

# In[2]:


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


# In[4]:


# Load dataset
movies_full = pd.read_csv("/Users/ankitadeshmukh/Desktop/SJSU/Academic/Fall22/CMPE257/Project/Dataset/serendipity-sac2018/movies.csv", on_bad_lines='skip')
movies_full.head()


# In[5]:


# Drop unnecessary columns
cols_to_drop = ['imdbId', 'tmdbId']
movies_full.drop(cols_to_drop, axis = 1, inplace = True)
movies_full.head()


# In[6]:


# Find numerical colums
movies_full.select_dtypes(exclude=['object']).columns.tolist()
# Find categorical colums
movies_full.select_dtypes(include=['object']).columns.tolist()
# Check for missing values
movies_full.isnull().sum()
# Check for duplicate values
movies_full.duplicated().sum()


# #### 1 numerical columns and 5 categorical columns

# #### No duplicate observations but some missing values

# In[7]:


# Since our dataset is large, removing the rows with missing values won't hurt.
movies_full.dropna(inplace=True)
# Check for missing values again
movies_full.isnull().sum()


# In[8]:


# Find the dimensions of this dataset
movies_full.shape


# ### Now we have 43018 unique movies in our dataset with 6 features

# ### Let's use one-hot-encoding to find the movie genres

# In[9]:


seperated_ratings = []

for col in movies_full["genres"]:
    try:
        col = col.split(",")
    except:
        col = ["Null"]
    seperated_ratings.append(col)

data = {'Seperated_genres': seperated_ratings}
test = pd.DataFrame(data)

one_hot = MultiLabelBinarizer()
res = one_hot.fit_transform(seperated_ratings)
classes = one_hot.classes_

df = pd.DataFrame(res, columns=classes)

movies_full = movies_full.drop('genres', axis=1)
movies_full = movies_full.join(df)
movies_full.head()


# In[10]:


movies_full.shape


# ### Now we have additional 18 features comapred to previous data which means we have 18 movie genres.

# In[11]:


movies_full.columns


# In[12]:


movies_full.describe()


# In[13]:


x={}
for i in movies_full.columns[5:23]:
    x[i]=movies_full[i].value_counts()[1]
    print("{}    \t\t\t\t{}".format(i,x[i]))

plt.bar(height=x.values(),x=x.keys())
plt.xticks(rotation=90)
plt.show()


# ### Most of the movies from the dataset belong to Drama genre followed by Comedy.

# In[3]:


# Load training dataset which contains the ratings for movies by different users
training_full = pd.read_csv("/Users/ankitadeshmukh/Desktop/SJSU/Academic/Fall22/CMPE257/Project/Dataset/serendipity-sac2018/training.csv")
training_full.head()


# In[4]:


# Drop unnecessary columns
cols_to_drop = ['timestamp']
training_full.drop(cols_to_drop, axis = 1, inplace = True)
training_full.head()


# In[16]:


# Find numerical colums
training_full.select_dtypes(exclude=['object']).columns.tolist()
# Find categorical colums
training_full.select_dtypes(include=['object']).columns.tolist()
# Check for missing values
training_full.isnull().sum()
# Check for duplicate values
training_full.duplicated().sum()


# In[5]:


training_full.shape


# #### 3 numerical columns and 0 categorical columns

# #### No missing values and no duplicate observations

# In[29]:


# Merge movies and training file based on movieId
movie_ratings_df = pd.merge(training_full, movies_full, on='movieId')
movie_ratings_df.head()


# In[30]:


#Extracting the year from the Title
movie_ratings_df['Year'] = movie_ratings_df['title'].str.extract('.*\((.*)\).*',expand = False)
movie_ratings_df.head()


# In[31]:


#Ploting a Graph with number of Movies each Year corresponding to its Year
plt.plot(movie_ratings_df.groupby('Year').title.count())
plt.show()
a=movie_ratings_df.groupby('Year').title.count()
print('Max No.of Movies Relesed =',a.max())
for i in a.index:
    if a[i] == a.max():
        print('Year =',i)
a.describe()


# ### Now we know that the maximum number of movies were in 2009 with count = 443747. On an average, 59120 movies are released every year.

# In[32]:


avg_rating_df = movie_ratings_df.groupby(['movieId']).agg (avg_rating = ('rating', 'mean'))
movie_ratings_df = pd.merge(movie_ratings_df, avg_rating_df, how='outer', on='movieId')
movie_ratings_df.head()


# In[33]:


count_df = movie_ratings_df.groupby(['movieId']).agg (user_count = ('userId', 'count'))
movie_ratings_df = pd.merge(movie_ratings_df, count_df, how='outer', on='movieId')
movie_ratings_df.head()


# In[38]:


new_df = movie_ratings_df[['userId', 'movieId', 'title', 'rating', 'avg_rating' ,'user_count']]
new_df.head()


# In[39]:


new_df.sort_values(['user_count', 'avg_rating'],ascending=False)


# ### Movies with the highest ratings

# In[40]:


# selecting rows based on condition 
rslt_df = new_df.loc[new_df['avg_rating'] == 5.0].sort_values(['user_count'],ascending=False)
rslt_df


# ### Most of the movies which are rated 5 stars are only rated by 1 or 2 people.

# ### Top-10 most watched movies

# In[41]:


rating_count = movie_ratings_df.groupby('title')['user_count']
rating_count = rating_count.count().sort_values(ascending=False)
rating_count[:10]


# ### The most watched movie from our dataset is "The Matrix" with 42120 views. It is also the highest rated movie considering user ratings and viewer count.

# ### Let's find out how different users rated "The Matrix".

# In[42]:


plt.figure(figsize=(8,6))
movies_grouped = movie_ratings_df.groupby('title')
the_matrix = movies_grouped.get_group('Matrix, The (1999)')
the_matrix['rating'].hist()
plt.title('User rating of the movie “The Matrix”')
plt.xlabel('Rating')
plt.ylabel('Number of Users')

plt.show()


# ### Let's see which user voted for the most number of movies.

# In[44]:


user_rating_df = movie_ratings_df.groupby(['userId']).agg (avg_user_rating = ('rating', 'mean'))
user_count_df = movie_ratings_df.groupby(['userId']).agg (avg_user_count = ('rating', 'count'))
user_ratings_df = pd.merge(movie_ratings_df, user_rating_df, how='outer', on='userId')
user_ratings_df = pd.merge(user_ratings_df, user_count_df, how='outer', on='userId')
user_ratings_df.head()


# In[47]:


# new_user_df = user_ratings_df.drop(['rating', 'releaseDate', 'directedBy', 'starring', 'genres', 'Year', 'user_count'],axis = 1)
# new_user_df.sort_values(by = "avg_user_count", ascending=False).head(10)

new_user_df = user_ratings_df[['userId', 'movieId', 'title', 'rating', 'avg_rating', 'avg_user_rating','user_count', 'avg_user_count']]
new_user_df.sort_values(by = "avg_user_count", ascending=False).head(10)
new_user_df.head()


# ### Top-10 users who voted the most number of movies

# In[48]:


rating_count = new_user_df.groupby('userId')['avg_user_count']
rating_count = rating_count.count().sort_values(ascending=False)
rating_count[:10]


# ### Rating pattern of User 148071 

# In[49]:


plt.figure(figsize=(8,6))
users_grouped = user_ratings_df.groupby('userId')
user_148071 = users_grouped.get_group(148071)
user_148071['rating'].hist()
plt.title('User rating for different movies')
plt.xlabel('Rating')
plt.ylabel('No of movies rated bu User 148071')

plt.show()


# ### There are other files like tags.csv, tag_genome.csv, answers.csv and we might use some of those files as metadata for content-based recommendations and latent-matrix factorization.

# In[4]:


model_list = ['User-Based CF', 'Item-Based CF', 'Content-Based CF', 'Latent Factor-SVD', 'Latent Factor-SVDpp']
mae_list = [1.23, 2.43, 4.03, 0.68, 0.67]


plt.figure(figsize=(12,8))
plt.bar(height = mae_list, x = model_list, color = 'blue')
plt.xticks(rotation=90)
plt.title('Performance Comparison')
plt.xlabel('Recommendation Algorithms')
plt.ylabel('Mean Absolute Error')
plt.show()


# In[ ]:
