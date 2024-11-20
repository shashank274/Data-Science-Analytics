import pandas as pd
import numpy as np
from scipy import stats

url = "https://archive.ics.uci.edu/static/public/352/data.csv"
df = pd.read_csv(url)
print(df.head())

# Drop rows with missing values
df.dropna(inplace=True)
numeric_columns = ['Quantity', 'UnitPrice', 'CustomerID']
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Handling duplicates
df.drop_duplicates(inplace=True)
df

#--------------------------------------------------------------------------------------------------------------#

import matplotlib.pyplot as plt
from scipy import stats

# Assuming you already have your DataFrame 'df' with relevant columns
numeric_columns = ['Quantity', 'UnitPrice', 'CustomerID']

# Calculate z-scores
z_scores = stats.zscore(df[numeric_columns])
abs_z_scores = np.abs(z_scores)

# Filter out rows where all z-scores are within 3 standard deviations
filtered_entries = (abs_z_scores < 3).all(axis=1)
df_no_outliers = df[filtered_entries]

# Now 'df_no_outliers' contains your data without outliers
print(df_no_outliers.head())

plt.scatter(range(len(df)), df['UnitPrice'])
plt.title('Visualization of Data Points with Outliers')
plt.xlabel('Indexing of Data Points')
plt.ylabel('Unit Price')
plt.show()

#Quantity
plt.scatter(range(len(df)), df['Quantity'])
plt.title('Visualization of Data Points with Outliers (Quantity)')
plt.xlabel('Indexing of Data Points')
plt.ylabel('Quantity')
plt.show()

#--------------------------------------------------------------------------------------------------------------#
import pandas as pd
from sklearn.cluster import KMeans


# Calculate Total Amount Spent
df['TotalAmountSpent'] = df['Quantity'] * df['UnitPrice']

# Calculate Purchase Frequency
df['PurchaseFrequency'] = df.groupby('CustomerID')['InvoiceNo'].transform('nunique')

# Calculate Recency
df['Recency'] = (pd.to_datetime(df['InvoiceDate']).max() - pd.to_datetime(df['InvoiceDate'])).dt.days

# Calculate Average Unit Price
df['AvgUnitPrice'] = df.groupby('CustomerID')['UnitPrice'].transform('mean')

# Country-Based
country_features = pd.get_dummies(df['Country'], prefix='Country')
df = pd.concat([df, country_features], axis=1)

# Calculate Total Quantity Purchased
df['TotalQuantityPurchased'] = df['Quantity'].sum()

# Calculate Average Quantity Per Invoice
df['AvgQuantityPerInvoice'] = df.groupby('InvoiceNo')['Quantity'].transform('mean')

#Calculate Monetary value
df['MonetaryValue'] = df['Quantity'] * df['UnitPrice']

#--------------------------------------------------------------------------------------------------------------#

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Assuming 'df' is your DataFrame with the relevant features
# Define the feature columns for scaling
feature_columns = [
    'TotalAmountSpent',
    'PurchaseFrequency',
    'Recency',
    'AvgUnitPrice',
    'TotalQuantityPurchased',
    'AvgQuantityPerInvoice'
]

# Extract features
X = df[feature_columns]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dimensionality Reduction using PCA
# Initialize PCA (you can choose the number of components based on your needs)
pca = PCA(n_components=2)  # You can also choose 3 or more components if needed
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame for PCA components
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])

# Add PCA components to the original DataFrame (optional)
df = pd.concat([df, pca_df], axis=1)

# Print the resulting DataFrame with PCA components
print(df.head())

#--------------------------------------------------------------------------------------------------------------#

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Assuming you have a DataFrame 'df' with relevant columns
features_for_clustering = df[['TotalAmountSpent', 'PurchaseFrequency']]

# Standardize the features (optional but recommended)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_for_clustering)

# Now 'scaled_features' contains your standardized features
print(scaled_features[:5])  # Print the first 5 rows as an example

#--------------------------------------------------------------------------------------------------------------#

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



# Determine the optimal number of clusters (k) using the elbow method
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, random_state=0)
    kmeans.fit(df[['TotalAmountSpent', 'PurchaseFrequency', 'Recency', 'AvgUnitPrice', 'TotalQuantityPurchased', 'AvgQuantityPerInvoice']])
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS (Inertia)')
plt.show()

# Based on the elbow method plot, choose the optimal k (number of clusters)
optimal_k = 3

# Perform K-means clustering with the chosen k
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, random_state=0)
clusters = kmeans.fit_predict(df[['TotalAmountSpent', 'PurchaseFrequency', 'Recency', 'AvgUnitPrice', 'TotalQuantityPurchased', 'AvgQuantityPerInvoice']])

# Add the cluster labels to the original DataFrame
df['CustomerSegment'] = clusters

# Create a scatter plot
plt.scatter(df['TotalAmountSpent'], df['PurchaseFrequency'], c=clusters, cmap='viridis')
plt.title(f'K-means Clustering (k = {k})')
plt.xlabel('Total Amount Spent')
plt.ylabel('Purchase Frequency')
plt.show()


#--------------------------------------------------------------------------------------------------------------#

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Define feature columns
feature_columns = [
    'TotalAmountSpent',
    'PurchaseFrequency',
    'Recency',
    'AvgUnitPrice',
    'TotalQuantityPurchased',
    'AvgQuantityPerInvoice'
]

# Extract features
X = df[feature_columns]

# Initialize PCA with 3 components
pca = PCA(n_components=3)

# Fit and transform the data
X_pca = pca.fit_transform(X)

# Add PCA components to your DataFrame
df['PC1'] = X_pca[:, 0]
df['PC2'] = X_pca[:, 1]
df['PC3'] = X_pca[:, 2]

# Scatter plot for the first two components
plt.figure(figsize=(10, 6))

# Ensure optimal_k is defined
optimal_k = df['CustomerSegment'].nunique()

for cluster_id in range(optimal_k):
    cluster_data = df[df['CustomerSegment'] == cluster_id]
    plt.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'Cluster {cluster_id}')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA-based Clustering Visualization (PC1 vs PC2)')
plt.legend()
plt.grid(True)
plt.show()

# Scatter plot for the second and third components
plt.figure(figsize=(10, 6))

for cluster_id in range(optimal_k):
    cluster_data = df[df['CustomerSegment'] == cluster_id]
    plt.scatter(cluster_data['PC2'], cluster_data['PC3'], label=f'Cluster {cluster_id}')

plt.xlabel('Principal Component 2')
plt.ylabel('Principal Component 3')
plt.title('PCA-based Clustering Visualization (PC2 vs PC3)')
plt.legend()
plt.grid(True)
plt.show()

#--------------------------------------------------------------------------------------------------------------#

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


feature_columns = [
    'TotalAmountSpent',
    'PurchaseFrequency',
    'Recency',
    'AvgUnitPrice',
    'TotalQuantityPurchased',
    'AvgQuantityPerInvoice'
]

X = df[feature_columns]

# Initialize PCA with 2 components
pca = PCA(n_components=2)

# Fit and transform the data
X_pca = pca.fit_transform(X)

# Add PCA components to your DataFrame
df['PC1'] = X_pca[:, 0]
df['PC2'] = X_pca[:, 1]

#scatter plot
plt.figure(figsize=(10, 6))


for cluster_id in range(optimal_k):
    cluster_data = df[df['CustomerSegment'] == cluster_id]
    plt.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'Cluster {cluster_id}')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA-based Clustering Visualization')
plt.legend()
plt.grid(True)
plt.show()

#--------------------------------------------------------------------------------------------------------------#

from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import cdist


cluster_labels = df['CustomerSegment']

# Calculate silhouette score
silhouette_avg = silhouette_score(X, cluster_labels)
print(f"Silhouette Score: {silhouette_avg:.4f}")

# Calculate Davies–Bouldin index
db_index = davies_bouldin_score(X, cluster_labels)
print(f"Davies–Bouldin Index: {db_index:.4f}")

# Calculate cohesion (within-cluster distance)
inertia = kmeans.inertia_

# Calculate separation (between-cluster distance)
centroids = kmeans.cluster_centers_
distances = cdist(centroids, centroids, 'euclidean')
max_separation = distances.max()

print(f"Cohesion (Within-Cluster Distance): {inertia:.4f}")
print(f"Separation (Between-Cluster Distance): {max_separation:.4f}")

#--------------------------------------------------------------------------------------------------------------#

from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

# Sample 10% of the data
sample_indices = df.sample(frac=0.1, random_state=42).index
X_sample = X.loc[sample_indices]
cluster_labels = df.loc[sample_indices, 'CustomerSegment']

# Standardize the features
scaler = StandardScaler()
X_sample_scaled = scaler.fit_transform(X_sample)

# Refit KMeans on the sampled data
kmeans_sample = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_sample.fit(X_sample_scaled)
centroids_sample = kmeans_sample.cluster_centers_


# Calculate cohesion (within-cluster distance)
cohesion = np.sum(np.min(cdist(X_sample_scaled, centroids_sample, 'euclidean'), axis=1))
print(f"Cohesion (Within-Cluster Distance): {cohesion:.4f}")

# Calculate separation (between-cluster distance)
distances = cdist(centroids_sample, centroids_sample, 'euclidean')
np.fill_diagonal(distances, np.inf)
min_separation = distances.min()
print(f"Separation (Between-Cluster Distance): {min_separation:.4f}")

#--------------------------------------------------------------------------------------------------------------#

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score

n_samples = int(0.1 * len(X))
X_subset = X[:n_samples]

# Define potential K values (number of clusters)
range_n_clusters = [2, 3, 4, 5]

for n_clusters in range_n_clusters:
    # Initialize KMeans with the current value of n_clusters
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X_subset)

    # Calculate silhouette scores
    silhouette_avg = silhouette_score(X_subset, cluster_labels)
    print(f"Silhouette Score for {n_clusters} clusters: {silhouette_avg:.4f}")

    # Compute silhouette samples for each data point
    sample_silhouette_values = silhouette_samples(X_subset, cluster_labels)

    # Create a subplot for silhouette plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(X_subset)), sample_silhouette_values, marker='o')
    plt.xlabel("Sample Index")
    plt.ylabel("Silhouette Score")
    plt.title(f"Silhouette Scores for {n_clusters} Clusters (Subset of Data)")
    plt.grid(True)
    plt.show()

#--------------------------------------------------------------------------------------------------------------#


# Normalize ranks and calculate RFM score
rfm_df = df[['CustomerID', 'Recency', 'PurchaseFrequency', 'MonetaryValue']]
rfm_df['R_rank'] = rfm_df['Recency'].rank(ascending=False)
rfm_df['F_rank'] = rfm_df['PurchaseFrequency'].rank(ascending=True)
rfm_df['M_rank'] = rfm_df['MonetaryValue'].rank(ascending=True)
rfm_df['R_rank_norm'] = (rfm_df['R_rank'] / rfm_df['R_rank'].max()) * 100

# Calculate RFM score
rfm_df['RFM_Score'] = 0.15 * rfm_df['R_rank_norm'] + 0.28 * rfm_df['F_rank'] + 0.57 * rfm_df['M_rank']

# Print the resulting DataFrame
print(rfm_df.head())

#--------------------------------------------------------------------------------------------------------------#

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import numpy as np

# Load your data (assuming df is your DataFrame and 'InvoiceNo' and 'Description' are the columns of interest)
# For market basket analysis, we need to transform the data into a list of transactions
transactions = df.groupby(['InvoiceNo'])['Description'].apply(lambda x: list(x)).tolist()

# Use TransactionEncoder to convert transactions into a one-hot encoded DataFrame
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_transactions = pd.DataFrame(te_ary, columns=te.columns_)

# Apply Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(df_transactions, min_support=0.01, use_colnames=True)

# Generate association rules from the frequent itemsets
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Display the rules
print("Frequent Itemsets:")
print(frequent_itemsets.head())

print("\nAssociation Rules:")
print(rules.head())

# Optional: Visualize the rules
# Plot support vs. confidence for the rules, with color indicating lift value
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sc = plt.scatter(rules['support'], rules['confidence'], c=rules['lift'], cmap='viridis', alpha=0.7)
plt.colorbar(sc, label='Lift')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Support vs Confidence for Association Rules')

# Plot support vs. lift for the rules, with color indicating confidence value
plt.subplot(1, 2, 2)
sc = plt.scatter(rules['support'], rules['lift'], c=rules['confidence'], cmap='plasma', alpha=0.7)
plt.colorbar(sc, label='Confidence')
plt.xlabel('Support')
plt.ylabel('Lift')
plt.title('Support vs Lift for Association Rules')

plt.tight_layout()
plt.show()

#--------------------------------------------------------------------------------------------------------------#

