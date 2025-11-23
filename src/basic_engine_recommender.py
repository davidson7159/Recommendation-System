import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# 1. LOAD THE DATA
# We use 'low_memory = False' because the dataset might be large/mixed types

print("Loading data ...")
try:
    df = pd.read_csv('data/raw/Anime.csv.zip')
except FileNotFoundError:
    # Fallback if running this from a different directory
    df = pd.read_csv('~/Documents/Projects/Recommendation-system/data/raw/Anime.csv.zip')


# 2. PREPROCESSING
print("Cleaning data ...")
# Let's fill missing values with empty strings so our math doesn't break
df['Synopsis'] = df['Synopsis'].fillna('')
df['Genres'] = df['Genres'].fillna('')

# Create a 'soup' of text. This combines the plot and the genre.
# We'll add the genre multiple times (*2) to give it more weight.
# For example, if an anime is "Action", we want that to matter more than a random word in the plot
df['soup'] = df['Synopsis'] + " " + (df['Genres'] + " ") * 2


# 3. VECTORIZATION
print("Converting text to numbers (TF-IDF)...")
tfidf = TfidfVectorizer(stop_words = 'english')

# Construct the TF-IDF matrix
# This creates a giant matrix where rows are Anime and columns are words
tfidf_matrix = tfidf.fit_transform(df['soup'])


# 4. SIMILARITY CALCULATION
print("Calculating similarity scores ...")
# linear_kernel is a faster implementation of cosine similarity for this kind of data
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# 5. RECOMMENDATION FUNCTION
indices = pd.Series(df.index, index=df['Title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    # Check if title exists
    if title not in indices:
        return f"Error: Anime '{title}' not found in the dataset."
    # Get the index of the anime that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores of all anime with that anime
    # This returns a list of (index, score) tuples
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the anime based on the similarity scores (highest first)
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar anime
    # We skip [0] because that is the anime itself (score = 1.0)
    sim_scores = sim_scores[1:11]

    # Get the anime indices
    anime_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar titles
    return df['Title'].iloc[anime_indices]


# --- TEST IT OUT ---
if __name__ == "__main__":
    test_anime = "Shingeki no Kyojin"
    print(f"\nRecommending anime similar to: {test_anime}")
    print("-" * 50)
    recommendations = get_recommendations(test_anime)
    print(recommendations)
