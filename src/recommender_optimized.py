import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# 1. LOAD DATA
def load_data():
    print("Loading data ...")
    try:
        df = pd.read_csv("data/raw/Anime.csv.zip")
    except FileNotFoundError:
        # Fallback if running this from a different directory
        df = pd.read_csv('~/Documents/Projects/Recommendation-system/data/raw/Anime.csv.zip')
    return df


# 2. CLEAN & PREPROCESS
def create_soup(x):
    # We combine multiple columns to create a rich "metadata soup"
    # We replace 'Unknown' with empty strings so it doesn't affect the model

    genres = x['Genres'] if x['Genres'] != 'Unknown' else ''
    themes = x['Themes'] if x['Themes'] != 'Unknown' else ''
    studios = x['Studios'] if x['Studios'] != 'Unknown' else ''
    synopsis = x['Synopsis'] if isinstance(x['Synopsis'], str) else ''

    return f"{synopsis} {genres} {themes} {studios}"

def prepare_data(df):
    print("Preparing data features ...")

    # Fill actual NaNs first
    df= df.fillna('')

    # Create the 'soup' column
    df['soup'] = df.apply(create_soup, axis=1)

    return df


# 3. BUILD MODEL
def train_model(df):
    print("Training model (TF-IDF) ...")
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['soup'])

    print("Calculating cosine similarity ...")
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim


# 4. SEARCH HELPER
def find_anime_index(title, df):
    # Try finding exact match in Title (Japanese/Romaji)
    if title in df['Title'].values:
        return df[df['Title'] == title].index[0]

    # Try finding exact match in English column
    if title in df['English'].values:
        return df[df['English'] == title].index[0]

    return None


# 5. RECOMMENDATION
def get_recommendations(title, df, cosine_sim):
    idx = find_anime_index(title, df)

    if idx is None:
        return [f"Anime '{title}' not found. Try the Japanese name?"]

    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort
    sim_scores = sorted(sim_scores, key = lambda x:x[1], reverse = True)

    # Top 10 (excluding self)
    sim_scores = sim_scores[1:11]
    anime_indices = [i[0] for i in sim_scores]

    # Return Title and English name for clarity
    return df[['Title', 'English']].iloc[anime_indices]

if __name__ == '__main__':
    # EXECUTION PIPELINE
    df = load_data()
    df = prepare_data(df)

    # Train
    cosine_sim = train_model(df)

    # Test with an English title now
    test_title = 'Attack on Titan'
    print(f"\n--- Recommendations for {test_title} ---")
    results = get_recommendations(test_title, df, cosine_sim)
    print(results)
    
