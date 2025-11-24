# ⛩️ Anime Recommendation Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Prototype-green)

A Content-Based Recommendation System that suggests anime based on plot summaries, genres, and production studios. Built with **Python**, **Scikit-Learn**, and **Streamlit**.

## 📖 Overview

Finding a new anime to watch can be overwhelming given the thousands of options available. This project solves the "Cold Start" problem by recommending anime similar to a user's favorite, analyzing the **content** (Synopsis, Genres, Themes) rather than user ratings.

It uses **Natural Language Processing (NLP)** techniques to vectorize text data and calculates similarity scores to find the closest matches.

## ✨ Features

- **Robust Search:** Handles English titles ("Attack on Titan"), Japanese titles ("Shingeki no Kyojin"), and partial matches.
- **Content-Based Filtering:** Recommends items based on intrinsic metadata (Plot, Genre, Studio).
- **Interactive UI:** A clean web interface built with Streamlit.
- **Optimized Performance:** Uses on-demand similarity calculation to minimize RAM usage.

## 🛠️ Tech Stack

- **Python**: Core logic.
- **Pandas**: Data manipulation and cleaning.
- **Scikit-Learn**: TF-IDF Vectorization and Cosine Similarity calculation.
- **Streamlit**: Web application framework.

## 📂 Project Structure

```text
Recommendation-system-project/
├── data/
│   └── raw/               # Contains Anime.csv
├── src/
│   ├── st_recommender_app.py         # Streamlit frontend application
|   |── basic_engine_recommender.py   # basci engine - baseline
|   |── second_engine_recommender.py  # improvement of the basic engine
|   |── main_recommender_terminal.py  # to run locally in the terminal
│   └── recommender_functions.py     # Backend logic (Data loading, TF-IDF, Search)
├── .gitignore             # Files to ignore (data, cache)
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

## 🚀 How to Run Locally
Clone the repository

Bash

git clone [https://github.com/davidson7159/Recommendation-System.git](https://github.com/davidson7159/Recommendation-System.git)
cd Recommendation-System
Install dependencies

Bash

pip install -r requirements.txt
Run the Streamlit App

Bash

cd src
streamlit run app.py
Enjoy! The app will open automatically in your browser at http://localhost:8501.

🧠 How It Works (The Data Science Part)
Text Preprocessing ("The Soup"): We combine critical metadata into a single string for each anime: Synopsis + Genres + Themes + Studios. Note: Genres are weighted (repeated) to prioritize category matching over plot keywords.

TF-IDF Vectorization: The textual data is converted into numerical vectors using Term Frequency-Inverse Document Frequency. This highlights unique words in a synopsis while ignoring common filler words.

Cosine Similarity: We calculate the cosine of the angle between the user's selected anime vector and all other anime vectors.

Score closer to 1: The anime are very similar.

Score closer to 0: The anime are unrelated.

🔜 Future Improvements
[ ] Add filters for "TV Show", "Movie", or "OVA".

[ ] Implement Hybrid Filtering (combining user ratings with content).

[ ] Deploy the application to Streamlit Cloud.

🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.

Created by Adrien Davidson
