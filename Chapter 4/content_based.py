import pandas as pd
import numpy as np

# Import data from clean file
df = pd.read_csv('tehmeerali/datasets/metadata_clean1/metadata_clean1.csv')


df.head()


# Import the original file
orig_df = pd.read_csv('tehmeerali/datasets/metadata_clean1/movies_metadata.csv', low_memory=False)

# Add the userful features into the cleaned dataframe
df['overview'], df['id'] = orig_df['overview'], orig_df['id']

df.head()


# Import IfIdVectorizer from the scikit-learn library
from sklearn.feature_extraction.text import TfidfVectorizer


# Define a TD-IDF Vectorizer Object. Remove all English Stop Words
tfidf = TfidfVectorizer(stop_words='english')

# Replace NaN with an empty string
df['overview'] = df['overview'].fillna('')

# Construct the required TF-IDF matrix by applying the fit_transform method on the overview feature
tfidf_matrix = tfidf.fit_transform(df['overview'])

# Output the shape of tfdif_matrix
tfidf_matrix.shape


# Import linear_kernal to compute the dot product
from sklearn.metrics.pairwise import linear_kernel


# Compute the consine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# Construct a reverse mapping of indices and movie titles, and drop duplicate titles, if any
indices = pd.Series(df.index, index=df['title']).drop_duplicates()


# Function that takes in movie title as input and gives recommendations
def content_recommender(title, cosine_sim=cosine_sim, df=df, indices=indices):
    # Obtain the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    # And convert it into a list of tuples as described above
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the cosine similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies. Ignore the first movie.
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df['title'].iloc[movie_indices]


#Get recommendations for The Lion King
content_recommender('The Lion King')
