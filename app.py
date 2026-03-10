import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movies data
movies_list = pickle.load(open('movies.pkl', 'rb'))

# Calculate similarity directly
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies_list['tags']).toarray()
similarity = cosine_similarity(vectors)

# Fetch poster from TMDB API
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=cded3c2444e943d4b491b8de1df1885a"
        data = requests.get(url, timeout=5).json()
        poster_path = data['poster_path']
        full_path = "https://image.tmdb.org/t/p/w500" + poster_path
        return full_path
    except:
        return "https://via.placeholder.com/500x750?text=No+Poster"

# Recommendation function
def recommend(movie):
    movie_index = movies_list[movies_list['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_sorted = sorted(list(enumerate(distances)),
                         reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movies_sorted:
        movie_id = movies_list.iloc[i[0]].movie_id
        recommended_movies.append(movies_list.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_posters

# Streamlit UI
st.title('🎬 Movie Recommendation System')

selected_movie = st.selectbox(
    'Select a movie',
    movies_list['title'].values
)

if st.button('Recommend'):
    names, posters = recommend(selected_movie)

    col1, col2, col3, col4, col5 = st.columns(5)
    cols = [col1, col2, col3, col4, col5]

    for idx, col in enumerate(cols):
        with col:
            st.text(names[idx])
            st.image(posters[idx])
# import streamlit as st
# import pickle
# import pandas as pd
# import requests

# # Load saved data
# movies_list = pickle.load(open('movies.pkl', 'rb'))
# similarity = pickle.load(open('similarity.pkl', 'rb'))

# # Fetch poster from TMDB API
# def fetch_poster(movie_id):
#     try:
#         url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=cded3c2444e943d4b491b8de1df1885a"
#         data = requests.get(url, timeout=5).json()
#         poster_path = data['poster_path']
#         full_path = "https://image.tmdb.org/t/p/w500" + poster_path
#         return full_path
#     except:
#         return "https://via.placeholder.com/500x750?text=No+Poster"
# # Recommendation function
# def recommend(movie):
#     movie_index = movies_list[movies_list['title'] == movie].index[0]
#     distances = similarity[movie_index]
#     movies_list2 = sorted(list(enumerate(distances)),
#                          reverse=True, key=lambda x: x[1])[1:6]

#     recommended_movies = []
#     recommended_posters = []

#     for i in movies_list2:
#         movie_id = movies_list.iloc[i[0]].movie_id
#         recommended_movies.append(movies_list.iloc[i[0]].title)
#         recommended_posters.append(fetch_poster(movie_id))

#     return recommended_movies, recommended_posters

# # Streamlit UI
# st.title('🎬 Movie Recommendation System')

# selected_movie = st.selectbox(
#     'Select a movie',
#     movies_list['title'].values
# )

# if st.button('Recommend'):
#     names, posters = recommend(selected_movie)

#     col1, col2, col3, col4, col5 = st.columns(5)
#     cols = [col1, col2, col3, col4, col5]

#     for idx, col in enumerate(cols):
#         with col:
#             st.text(names[idx])
#             st.image(posters[idx])
# ```

# ---

# **Save this file inside your `movie-recommender` folder**

# ---

# **Get TMDB API Key:**

# 1. Go to **themoviedb.org**
# 2. Create free account
# 3. Go to **Settings → API**
# 4. Request API key
# 5. Copy the key
# 6. Replace **YOUR_API_KEY** in the code with your actual key

# ---

# **Then run the app:**

# 1. Open **Anaconda Prompt**
# 2. Navigate to your folder:
# ```
# cd Desktop/movie-recommender
# ```
# 3. Run:
# ```

# streamlit run app.py

