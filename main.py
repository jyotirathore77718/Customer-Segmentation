import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('customer_data.csv')

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Fill missing values or drop rows/columns with missing values
data = data.dropna()

# Encoding categorical variables if any (assuming 'Category' is a categorical column)
# data['Category'] = pd.get_dummies(data['Category'], drop_first=True)

# Normalize numerical features if needed
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Convert the scaled data back to a DataFrame
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

# Determine the optimal number of clusters using the Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

# Plot the results
plt.figure(figsize=(10, 8))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Apply K-means clustering with the optimal number of clusters
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)

# Display the first few rows with the cluster labels
print(data.head())

# Group by clusters and calculate the mean for each feature
cluster_summary = data.groupby('Cluster').mean()
print(cluster_summary)

# Visualize the clusters (example with a pairplot, assuming data has numerical features)
sns.pairplot(data, hue='Cluster', palette='Set1')
plt.show()

# Save the data with cluster labels to a new CSV file
data.to_csv('customer_segments.csv', index=False)
