import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Load the MovieLens dataset
columns = ['user_id', 'movie_id', 'rating', 'timestamp']
df = pd.read_csv(r"C:\Users\chand\OneDrive\Documents\PGM\Anurva_ML\movierecom\movierecom\ml-100k\ml-100k\u.data", sep='\t', names=columns)

# Load movie titles
movies = pd.read_csv(r"C:\Users\chand\OneDrive\Documents\PGM\Anurva_ML\movierecom\movierecom\ml-100k\ml-100k\u.item", sep='|', encoding='latin-1', header=None, usecols=[0, 1], names=['movie_id', 'title'])

# Merge dataframes on movie_id
df = pd.merge(df, movies, on='movie_id')

# Create a pivot table with users and movies as rows and columns
user_movie_ratings = df.pivot_table(index='user_id', columns='title', values='rating')

# Fill NaN values with 0
user_movie_ratings = user_movie_ratings.fillna(0)

# Transpose the matrix so that movies become rows and users become columns
movie_user_ratings = user_movie_ratings.transpose()

# Build a KNN model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(movie_user_ratings)

# Function to recommend movies
def recommend_movies(movie_title, n_neighbors=5):
    try:
        print(f"Searching for movie: {movie_title}")
        distances, indices = model_knn.kneighbors(movie_user_ratings.loc[movie_title].values.reshape(1, -1), n_neighbors=n_neighbors+1)
        print(f"Distances: {distances}")
        print(f"Indices: {indices}")
        recommended_movies = [movie_user_ratings.index[i] for i in indices.flatten()[1:]]
        print(f"Recommended movies: {recommended_movies}")
        return recommended_movies
    except KeyError:
        print(f"Movie {movie_title} not found in dataset.")
        return []

# GUI
class MovieRecommendationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Movie Recommendation System")

        # Movie Title Label and Entry
        ttk.Label(root, text="Enter a Movie Title:").grid(row=0, column=0, padx=10, pady=10)
        self.movie_entry = ttk.Entry(root, width=30)
        self.movie_entry.grid(row=0, column=1, padx=10, pady=10)

        # Recommendation Button
        ttk.Button(root, text="Get Recommendations", command=self.get_recommendations).grid(row=1, column=0, columnspan=2, pady=10)

        # Recommendation Listbox
        self.recommendation_listbox = tk.Listbox(root, height=10, width=50)
        self.recommendation_listbox.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

    def get_recommendations(self):
        movie_title = self.movie_entry.get()
        recommended_movies = recommend_movies(movie_title)
        self.display_recommendations(recommended_movies)

    def display_recommendations(self, recommended_movies):
        if recommended_movies:
            self.recommendation_listbox.delete(0, tk.END)
            for movie in recommended_movies:
                self.recommendation_listbox.insert(tk.END, movie)
        else:
            messagebox.showinfo("No Recommendations", "No recommendations found for the entered movie.")

# Create and run the GUI
root = tk.Tk()
app = MovieRecommendationApp(root)
root.mainloop()