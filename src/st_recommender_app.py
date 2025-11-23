# Fichier: src/app.py
import streamlit as st
import pandas as pd
# On importe notre propre module !
# Note : Cela suppose que les deux fichiers sont dans le même dossier 'src'
import recommender_functions as engine 

st.set_page_config(page_title="Recommandeur d'Anime", page_icon="⛩️", layout="wide")

# --- WRAPPERS POUR LE CACHE STREAMLIT ---
# On "enveloppe" les fonctions importées pour que Streamlit puisse les mémoriser
@st.cache_data
def get_data():
    return engine.load_and_clean_data()

@st.cache_resource
def get_trained_model(df):
    return engine.build_model(df)

# --- INTERFACE ---
st.title("⛩️ Moteur de Recommandation (Version Modularisée)")

# 1. Chargement
with st.spinner("Démarrage du moteur..."):
    df = get_data()
    tfidf_matrix = get_trained_model(df)

# 2. Recherche
search_list = pd.concat([df['Title'], df['English']]).unique()
search_list = [x for x in search_list if x and x != "Unknown"]

selected_name = st.selectbox("Rechercher :", [""] + sorted(list(search_list)))

# 3. Résultats
if selected_name:
    # On appelle la fonction de recherche de notre module importé
    idx = engine.robust_search(selected_name, df)
    
    if idx is not None:
        recs = engine.get_recommendations_from_index(idx, tfidf_matrix, df)
        
        st.success(f"Résultats pour : {selected_name}")
        st.dataframe(recs[['Title', 'English', 'Genres', 'Score']]) # Affichage simple tableau
    else:
        st.error("Anime introuvable.")
