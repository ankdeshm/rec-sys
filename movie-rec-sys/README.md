# CMPE257-MovieRecommenderSystem
## Project Title: Movie Recommender System

## Team Information:
Ankita Arvind Deshmukh - ankdeshm <br>
Ganesh S Tulshibagwale - Ganesh-S-Tulshibagwale <br>
Indranil Dutta - d1ndra <br>
Pranav Chellagurki - pranav4099 <br>

## About Data:
## Dataset Name: 
Investigating Serendipity in Recommender Systems Based on Real User Feedback
## Source: 
https://grouplens.org/datasets/serendipity-2018/
## Data Summary: 
GroupLens Research group at the University of Minnesota and the University of Jyväskylä conducted an experiment in MovieLens (http://movielens.org) where users were asked how serendipitous particular movies were to them. This dataset contains user answers to GroupLens’ questions and additional information, such as past ratings of these users, recommendations they received before replying to the survey and movie descriptions. The dataset was generated on January 15, 2018. The data are contained in the files ‘answers.csv’, ‘movies.csv’, ‘recommendations.csv’, ‘tag_genome.csv’, ‘tags.csv’ and ‘training.csv’. Overall, there are 10,000,000 ratings (2,150 ratings stored in `answers.csv` and 9,997,850 in ‘training.csv’).

## Problem Description: 
•	To find k-similar users to every user and k-similar items (movies) to every item in the dataset <br>
•	To create user profile and movie profile to identify similarities between these vectors for prediction  <br>
•	To analyze the effect of various movie features such as genres, actors, directors, release date (metadata/content) on the rating prediction <br>
•	To design a model which predicts the ratings for users based on user/item similarity and content <br>
•	To provide movie recommendation to users based on the predicted ratings and perform a qualitative comparison of different approaches <br>

## Potential Methods:
•	Similarity metrics such as Cosine, Raw Cosine, Pearson similarity coefficient etc. <br>
•	User-based collaborative-filtering using similarity among different users <br>
•	Item-based collaborative-filtering using similarity among different items (movies) <br>
•	Content-based recommendation system using feature vector for movies (user-item profile) <br>
•	Latent-matrix factorization-based recommendation system using other metadata <br>

## Preprocessing:
Following steps are performed in data-wrangling <br>
•	Remove unnecessary features that are not planned to be used such as timestamp, IMDB ID etc. <br>
•	Find dimensions and statistical summary (min, max, mean, median, range, count, etc.) of the dataset <br>
•	Check for missing values and handle them <br>
•	Check and duplicate observations and handle them <br>
•	Factor numerical and categorical columns <br>
•	One-hot encoding for categorical column - Movie Genre  <br>
•	Some visualization for movies and users <br>

## Challenges:
•	Dataset sampling - Possibility of missing out on relevant information <br>
•	Feature engineering - What attributes are irrelevant to the problem statement? <br>
•	Class imbalance - Are all the classes represented equally? <br>
•	Missing data - How do different imputation methods affect the model accuracy? <br>
•	Data splitting – How to ensure proportional user representation and reliable test and train dataset sizes? <br>

## References:
[1] Denis Kotkov, Joseph A. Konstan, Qian Zhao, and Jari Veijalainen. 2018. Investigating Serendipity in Recommender Systems Based on Real User Feedback. In Proceedings of SAC 2018: Symposium on Applied Computing , Pau, France, April 9–13, 2018 (SAC 2018), 10 pages. DOI: 10.1145/3167132.3167276 <br>
[2] Jesse Vig, Shilad Sen, and John Riedl. 2012. The Tag Genome: Encoding Community Knowledge to Support Novel Interaction. ACM Trans. Interact. Intell. Syst. 2, 3: 13:1–13:44. https://doi.org/10.1145/2362394.2362395 <br>






