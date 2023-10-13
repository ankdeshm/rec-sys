# Fashion_Recommendation_System

It is a challenging task to develop a fashion suggestion system due to its subjectivity and complexity. While collaborative filtering and content-based filtering are quite popular in conventional recommendation systems, they are quite ineffective in the fashion industry. Considering that the visual signals are a key feature in fashion analysis, this fashion recommendation system finds visial similarities among products, outfits and scenes to recommend fashion products to users. To find the image similarity, cosine similarity and Spotify's Annoy model is used on Pinterest's Shop The Look dataset.

The main parts of this project are as below:

## Image classification to assign labels to fashion products

![Image1](images/img_class.png)

## Image similarity computation among similar products using cosine distance metric 

![Image2 Image](images/cosine_prod.png)

## Image similarity computation among similar products using ANNOY 

![Image3](images/annoy_prod.png)


## Image similarity computation among similar outfits/scenes using ANNOY 

![Image4](images/annoy_outfit.png)

