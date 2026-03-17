from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

data = pd.read_csv("movies.csv")

similarity = cosine_similarity(data)

def recommend(movie_index):
    scores = list(enumerate(similarity[movie_index]))
    scores = sorted(scores, key=lambda x:x[1], reverse=True)
    return scores[1:5]