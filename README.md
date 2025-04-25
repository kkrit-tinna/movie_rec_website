# Movie Recommendation System part 2
A movie recommendation system based on cosine similarity that suggests films similar to ones you enjoy. This package offers web-based interfaces for finding movie recommendations.

## About Dataset
Two datasets required for this packages:
One is the original movie database without embedded, users can download it from kaggle [**Full TMDB Movies Dataset 2024**](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies/data)<br>
The TMDb (The Movie Database) is a comprehensive movie database that provides information about movies, including details like titles, ratings, release dates, revenue, genres, and much more.<br>
(This dataset contains a collection of 1,000,000 movies from the TMDB database, and it is updated daily.)
The other dataset is all relevant movies in the original database embedded and is available from kaggle [**imdb_dataset_embedded**](https://www.kaggle.com/datasets/shikristin/imdbdataset)

## Features
- **Content-Based Recommendations**: Find movies with similar characteristics using cosine similarity
- **Two Interface Options**: Choose between a web application or desktop GUI
- **Movie Search with Autocomplete**: Easily find movies with typeahead suggestions
- **Similarity Scores**: See how closely recommended movies match your selection
- **Trending Movies**: Discover popular movies when specific titles aren't found
- **Visual Movie Information**: View posters, ratings, release years, and runtime
- **Responsive Design**: Works on various screen sizes (web version)

## Prerequisites

- Python 3.6 or higher
- Required packages:
  - For web application: `flask`, `pandas`, `scikit-learn`
  - For GUI application: `PyQt5`, `pandas`, `scikit-learn`

## Installation

1. Clone or download this repository
2. Place your dataset files in the project directory:
   - `TMDB_movie_dataset_v11.csv`: Original movie dataset
   - `movie_rec_databse_2.csv`: Pre-embedded dataset for similarity calculations

3. Install required packages:
   ```bash
   # For web application
   pip install flask pandas scikit-learn
   
   ```

## Directory Structure

```
movie-recommender/
├── app.py                       # Web application backend (Flask)
├── templates/                   # Folder for web templates
│   └── index.html               # Web interface
├── TMDB_movie_dataset_v11.csv   # Original movie dataset
└── movie_rec_databse_2.csv      # Embedded dataset for similarity
└── README.md
```

### Web Application

1. Navigate to the project directory
2. Run the Flask application:
   ```bash
   python app.py
   ```
3. Open your web browser and go to:
   ```
   http://localhost:5000
   ```
4. Type a movie title, select from autocomplete suggestions, and click "Find Recommendations"


## How It Works

The recommendation system uses these key components:

1. **Data Processing**:
   - The original dataset contains movie metadata (title, rating, etc.)
   - The embedded dataset contains vector representations of movies for similarity calculations

2. **Cosine Similarity**:
   - When you search for a movie, the system finds its vector representation
   - It calculates the cosine similarity with all other movie vectors
   - Movies with the highest similarity scores are recommended

3. **User Interface**:
   - The web interface uses Flask for the backend and HTML/CSS/JavaScript for the frontend
   - The GUI interface uses PyQt5 to create a native desktop application

## Customization

### Web Application Customization
- Modify `templates/index.html` to change the appearance of the web interface
- Adjust the CSS styles within the HTML file for different colors and layouts
- Edit `app.py` to change backend behavior like the number of recommendations


## Troubleshooting

### Common Issues

- **FileNotFoundError**: Make sure your dataset files are in the correct location and named properly
- **ImportError**: Verify that you've installed all required packages
- **Memory Issues**: For large datasets, consider using smaller sample files for testing

### Memory Optimization

If your datasets are very large, you can modify the loading code to improve performance:

```python
# Load only necessary columns
df = pd.read_csv(ORIGINAL_DATASET_PATH, usecols=['title', 'release_date', 'vote_average', 'runtime', 'poster_path', 'overview'])
```

## Extending the System

Some ideas for extending the functionality:

- **Add User Accounts**: Allow users to save favorites and get personalized recommendations
- **Implement Collaborative Filtering**: Combine content-based with user preference data
- **Add External APIs**: Connect to movie databases for up-to-date information and posters
- **Improve Visualization**: Add charts or graphs to visualize movie similarities
- **Filter Options**: Add genre, year, or rating filters to narrow recommendations

## License
This project is provided as-is for educational and personal use.

## Acknowledgments

This project uses:
- Cosine similarity for recommendation calculations
- Flask for the web application backend
- Pandas and scikit-learn for data processing

---

For questions, issues, or contributions, please open an issue on the repository.


## Resources
web dev using Flask
@https://code.visualstudio.com/docs/python/tutorial-flask


## Author
Yifei Shi


## Version History
* 0.1
    * Initial Release

