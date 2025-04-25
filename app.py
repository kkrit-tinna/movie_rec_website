# how to create virtual environmenr on terminal

# 1. Open your terminal or command prompt.
# 2. Navigate to the directory where you want to create the virtual environment.
# 3. Run the following command to create a virtual environment named "venv":
# python -m venv venv
# 4. Activate the virtual environment:
# - On Windows:
# venv\Scripts\activate
# - On macOS and Linux:
# source venv/bin/activate
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Global variables to store the loaded dataframes
df = None
df_final = None

def load_datasets():
    """Load and prepare the movie datasets."""
    global df, df_final
    
    # Paths to datasets - update these to your actual file paths
    ORIGINAL_DATASET_PATH = 'TMDB_movie_dataset_v11.csv'
    EMBEDDED_DATASET_PATH = 'movie_rec_databse_2.csv'
    
    # Check if files exist
    if not os.path.exists(ORIGINAL_DATASET_PATH):
        raise FileNotFoundError(f"Original dataset file not found: {ORIGINAL_DATASET_PATH}")
    if not os.path.exists(EMBEDDED_DATASET_PATH):
        raise FileNotFoundError(f"Embedded dataset file not found: {EMBEDDED_DATASET_PATH}")
    
    # Load the embedded dataframe for cosine similarity calculation
    df_final = pd.read_csv(EMBEDDED_DATASET_PATH)
    df_final.set_index('title', inplace=True)
    # Standardize movie names in the final dataframe
    df_final.index = df_final.index.str.strip().str.lower().str.replace(' ', '')
    
    # Load the original dataset
    df = pd.read_csv(ORIGINAL_DATASET_PATH)
    # Clean up the original dataset
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    # Standardize movie names in the original dataset
    df['title_normalized'] = df['title'].str.strip().str.lower().str.replace(' ', '')
    
    print(f"Loaded {len(df)} movies from original dataset and {len(df_final)} from embedded dataset")

def get_recommendation(movie_name, n=10):
    """
    Get movie recommendations based on similarity.
    
    Args:
        movie_name: Name of the movie to find recommendations for
        n: Number of recommendations to return
        
    Returns:
        Dictionary with recommendations or trending movies
    """
    # Normalize the movie name
    movie_name = movie_name.strip().lower().replace(' ', '')
    
    # Check if movie exists in the embedded dataset
    if movie_name not in df_final.index:
        # Return trending movies if movie not found
        trending_movies = get_trending_movies(n)
        return {
            "found": False,
            "message": f"Movie not found. Here are the top {n} trending movies for inspiration:",
            "recommendations": trending_movies
        }
    
    # Get the embedding for the requested movie
    new_df = df_final.loc[[movie_name]]
    
    # Remove rows with NaN values from the other movies
    df_other = df_final.loc[df_final.index != movie_name, :].dropna()
    
    # Get the titles of the other movies
    df_titles = df_other.index
    
    # Calculate cosine similarity
    cosine_sim_matrix = cosine_similarity(new_df, df_other)
    cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=[movie_name], columns=df_titles)
    
    # Get the top n most similar movies
    top_n_similar = cosine_sim_df.T.sort_values(by=movie_name, ascending=False).head(n)
    
    # Get the original titles (with proper capitalization)
    similar_titles = list(top_n_similar.index)
    original_titles_mask = df['title_normalized'].isin(similar_titles)
    top_n_movies = df[original_titles_mask].copy()
    
    # Add similarity scores
    top_n_movies['similarity_score'] = top_n_movies['title_normalized'].map(
        top_n_similar[movie_name]
    )
    
    # Sort by similarity score
    top_n_movies = top_n_movies.sort_values('similarity_score', ascending=False)
    
    # Convert to list of dictionaries for JSON response
    recommendations = []
    for _, movie in top_n_movies.iterrows():
        # Include only needed fields to reduce response size
        movie_data = {
            'title': movie['title'],
            'year': int(movie['release_date'].split('-')[0]) if pd.notna(movie['release_date']) else None,
            'score': float(movie['similarity_score']),
            'rating': float(movie['vote_average']) if pd.notna(movie['vote_average']) else None,
            'runtime': int(movie['runtime']) if pd.notna(movie['runtime']) else None,
        }
        
        # Add poster path if available
        if 'poster_path' in movie and pd.notna(movie['poster_path']):
            movie_data['poster_path'] = movie['poster_path']
        
        # Add overview if available
        if 'overview' in movie and pd.notna(movie['overview']):
            movie_data['overview'] = movie['overview']
            
        recommendations.append(movie_data)
    
    return {
        "found": True,
        "query": movie_name,
        "recommendations": recommendations
    }

def get_trending_movies(n=10):
    """Get trending movies as fallback."""
    # Use the most popular movies from the original dataset
    if 'popularity' in df.columns:
        trending = df.sort_values('popularity', ascending=False).head(n)
    else:
        trending = df.head(n)
    
    # Convert to list of dictionaries
    trending_movies = []
    for _, movie in trending.iterrows():
        movie_data = {
            'title': movie['title'],
            'year': int(movie['release_date'].split('-')[0]) if pd.notna(movie['release_date']) else None,
            'rating': float(movie['vote_average']) if pd.notna(movie['vote_average']) else None,
            'runtime': int(movie['runtime']) if pd.notna(movie['runtime']) else None,
        }
        
        if 'poster_path' in movie and pd.notna(movie['poster_path']):
            movie_data['poster_path'] = movie['poster_path']
            
        if 'overview' in movie and pd.notna(movie['overview']):
            movie_data['overview'] = movie['overview']
            
        trending_movies.append(movie_data)
    
    return trending_movies

def search_movies(query, limit=10):
    """Search for movies by title fragment."""
    if not query:
        return []
        
    # Normalize query
    query_normalized = query.strip().lower()
    
    # Find movies that contain the query string
    matches = df[df['title'].str.lower().str.contains(query_normalized)].head(limit)
    
    # Convert to list of dictionaries
    search_results = []
    for _, movie in matches.iterrows():
        movie_data = {
            'title': movie['title'],
            'year': int(movie['release_date'].split('-')[0]) if pd.notna(movie['release_date']) else None,
        }
        search_results.append(movie_data)
    
    return search_results

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/api/recommend', methods=['GET'])
def recommend():
    """API endpoint for recommendations by movie title."""
    movie_title = request.args.get('title', '')
    n = request.args.get('n', 10, type=int)
    
    if not movie_title:
        return jsonify({
            "error": "No movie title provided",
            "trending": get_trending_movies(n)
        }), 400
        
    # Limit n to a reasonable range
    n = max(1, min(20, n))
    
    # Get recommendations
    result = get_recommendation(movie_title, n)
    return jsonify(result)

@app.route('/api/search', methods=['GET'])
def search():
    """API endpoint for movie title search."""
    query = request.args.get('q', '')
    limit = request.args.get('limit', 10, type=int)
    
    # Limit to a reasonable range
    limit = max(1, min(20, limit))
    
    results = search_movies(query, limit)
    return jsonify({"results": results})

@app.route('/api/trending', methods=['GET'])
def trending():
    """API endpoint for trending movies."""
    n = request.args.get('n', 10, type=int)
    n = max(1, min(20, n))
    
    trending_movies = get_trending_movies(n)
    return jsonify({"trending": trending_movies})

if __name__ == '__main__':
    # Load datasets on startup
    load_datasets()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
    