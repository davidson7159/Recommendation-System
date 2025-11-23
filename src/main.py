import pandas as pd
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# --- CONFIGURATION ---
# Adjust this if your terminal supports colors (ANSI codes)
GREEN = "\033[92m"
BOLD = "\033[1m"
RESET = "\033[0m"
RED = "\033[91m"

def load_data():
    print(f"{BOLD}Loading data...{RESET}")
    try:
        df = pd.read_csv('data/raw/Anime.csv')
    except FileNotFoundError:
        try:
            df = pd.read_csv('~/Documents/Projects/Recommendation-system/data/raw/Anime.csv.zip')
        except:
            print(f"{RED}Error: Anime.csv not found.{RESET}")
            sys.exit(1)
            
    # Clean up "Unknown" values
    df = df.fillna('')
    return df

def train_engine(df):
    print(f"{BOLD}Training AI Model...{RESET}")
    
    # Create the "Soup" (Metadata combination)
    # We are careful to handle non-string data if any exists
    df['soup'] = df.apply(lambda x: f"{x['Synopsis']} {x['Genres']} {x['Themes']} {x['Studios']}", axis=1)
    
    # Vectorize
    tfidf = TfidfVectorizer(stop_words='english', min_df=2)
    tfidf_matrix = tfidf.fit_transform(df['soup'])
    
    return tfidf_matrix

def robust_search(user_input, df):
    """
    Smart search that finds anime even if you don't type the perfect Japanese title.
    """
    user_input = user_input.strip().lower()
    titles_lower = df['Title'].astype(str).str.lower()
    english_lower = df['English'].astype(str).str.lower()
    
    # 1. Exact Match
    if user_input in titles_lower.values:
        return titles_lower[titles_lower == user_input].index[0]
    if user_input in english_lower.values:
        return english_lower[english_lower == user_input].index[0]
        
    # 2. Partial Match (e.g. "Titan" in "Attack on Titan")
    mask = titles_lower.str.contains(user_input, regex=False) | english_lower.str.contains(user_input, regex=False)
    if mask.any():
        return mask.idxmax()
        
    return None

def get_recommendations(idx, tfidf_matrix, df):
    # Calculate similarity on demand
    query_vector = tfidf_matrix[idx]
    sim_scores = linear_kernel(query_vector, tfidf_matrix).flatten()
    
    # Get top 11 (including self)
    related_indices = sim_scores.argsort()[:-12:-1]
    
    # Exclude self
    related_indices = [i for i in related_indices if i != idx]
    
    return df.iloc[related_indices[:10]]

def main():
    # 1. Setup
    df = load_data()
    tfidf_matrix = train_engine(df)
    
    print(f"\n{GREEN}System Ready!{RESET}")
    print("Type an anime name to get recommendations.")
    print("Type 'exit' or 'quit' to stop.\n")
    
    # 2. Interactive Loop
    while True:
        user_input = input(f"{BOLD}>> Enter Anime Title: {RESET}")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        if not user_input.strip():
            continue
            
        # 3. Search
        idx = robust_search(user_input, df)
        
        if idx is not None:
            title = df.loc[idx, 'Title']
            english = df.loc[idx, 'English']
            display_title = f"{title} ({english})" if english else title
            
            print(f"\nFound match: {GREEN}{display_title}{RESET}")
            print("-" * 50)
            
            # 4. Recommend
            recs = get_recommendations(idx, tfidf_matrix, df)
            
            # Print results nicely
            for i, (_, row) in enumerate(recs.iterrows(), 1):
                r_title = row['Title']
                r_score = row['Score'] if row['Score'] != 'Unknown' else 'N/A'
                r_genres = row['Genres']
                print(f"{i}. {BOLD}{r_title}{RESET} [Score: {r_score}]")
                print(f"   Genres: {r_genres}")
            print("-" * 50 + "\n")
            
        else:
            print(f"{RED}Anime not found. Try a different spelling.{RESET}\n")

if __name__ == "__main__":
    main()
