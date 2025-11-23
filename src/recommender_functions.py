# Fichier: src/recommender.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def load_and_clean_data(path='~/Documents/Projects/Recommendation-system/data/raw/Anime.csv.zip'):
    # Gestion des chemins relatifs/absolus
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        # Essai de remonter d'un niveau si lancé depuis src/
        df = pd.read_csv(f'../{path}')
    
    return df.fillna('')

def build_model(df):
    """Crée la matrice TF-IDF"""
    # Création de la "Soup"
    df['soup'] = df['Synopsis'] + " " + (df['Genres'] + " ") * 2 + " " + df['Themes'] + " " + df['Studios']
    
    # Vectorisation
    tfidf = TfidfVectorizer(stop_words='english', min_df=2)
    tfidf_matrix = tfidf.fit_transform(df['soup'])
    
    return tfidf_matrix

def robust_search(user_input, df):
    """Recherche insensible à la casse et partielle"""
    user_input = user_input.strip().lower()
    titles_lower = df['Title'].str.lower()
    english_lower = df['English'].str.lower()
    
    # 1. Exact
    if user_input in titles_lower.values:
        return titles_lower[titles_lower == user_input].index[0]
    if user_input in english_lower.values:
        return english_lower[english_lower == user_input].index[0]
        
    # 2. Partiel
    mask = titles_lower.str.contains(user_input, regex=False) | english_lower.str.contains(user_input, regex=False)
    if mask.any():
        return mask.idxmax()
        
    return None

def get_recommendations_from_index(idx, tfidf_matrix, df):
    """Calcule les similarités pour un index donné"""
    query_vector = tfidf_matrix[idx]
    sim_scores = linear_kernel(query_vector, tfidf_matrix).flatten()
    related_indices = sim_scores.argsort()[:-12:-1]
    related_indices = [i for i in related_indices if i != idx]
    return df.iloc[related_indices[:10]]
