import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = 'spotify_dataset.csv'  # Update with your actual dataset path
spotify_data = pd.read_csv(file_path)

# Handle missing values
spotify_data[['track_name', 'track_artist', 'track_album_name']] = spotify_data[
    ['track_name', 'track_artist', 'track_album_name']
].fillna("Unknown")

# Numerical and Categorical Features
numerical_features = [
    'track_popularity', 'danceability', 'energy', 'key',
    'loudness', 'mode', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'
]

categorical_features = [
    'track_id', 'track_name', 'track_artist', 'track_album_id',
    'track_album_name', 'track_album_release_date', 'playlist_name',
    'playlist_id', 'playlist_genre', 'playlist_subgenre'
]

# Plot the distribution of numerical features
sns.set(style="whitegrid")
for feature in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(data=spotify_data, x=feature, kde=True, color="skyblue")
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

# Correlation Matrix
plt.figure(figsize=(12, 8))
corr_matrix = spotify_data[numerical_features].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Clustering Example
scaler = StandardScaler()
scaled_data = scaler.fit_transform(spotify_data[numerical_features])

kmeans = KMeans(n_clusters=5, random_state=42)
spotify_data['Cluster'] = kmeans.fit_predict(scaled_data)

# Plot clusters
plt.figure(figsize=(10, 6))
sns.countplot(x='Cluster', data=spotify_data, palette='viridis')
plt.title('Cluster Distribution')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.show()

print("Script execution complete. Modify the script further as needed.")
