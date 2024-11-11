
# K-Means
### Choosing the Right Number of Clusters
#### 1. Elbow Method / The Within Cluster Sum of Squares (WCSS)
For each cluster, calculate the squared distance of other points in the same cluster to the centroid and sum them. Calculate for each cluster.

Calculate the number of clusters for which increasing the number of clusters does not decrease WCSS.

![](https://imgs.search.brave.com/S375W7i4sbazwDDMbtKJfNz2efEl71MpaiLmVgjR470/rs:fit:500:0:0:0/g:ce/aHR0cHM6Ly9zMy5z/dGFja2FidXNlLmNv/bS9tZWRpYS9hcnRp/Y2xlcy9rLW1lYW5z/LWNsdXN0ZXJpbmct/d2l0aC10aGUtZWxi/b3ctbWV0aG9kLTEu/cG5n)

#### 2. The Average Silhouette Score
The silhouette coefficient $S(i)$ for a particular data point $i$ is a measure of how similar $i$ is to its own cluster compared to other clusters. The formula for calculating the silhouette coefficient is as follows:

$S(i) = \frac{b(i) - a(i)}{\max{a(i), b(i)}}$

Where:
- $S(i)$ is the silhouette coefficient for data point $i$.
- $a(i)$ is the average distance between $i$ and all the other data points in the same cluster as $i$ .
- $b(i)$ is the smallest average distance from $i$ to all clusters to which $i$ does not belong. In other words, it's the distance to the nearest cluster that $i$ is not a part of.

The silhouette coefficient $S(i)$ ranges between -1 and 1:
- $S(i)$ close to 1 indicates that the data point is well-clustered.
- $S(i)$ close to 0 indicates that the data point is on or very close to the decision boundary between two neighboring clusters.
- $S(i)$ close to -1 indicates that the data point might have been assigned to the wrong cluster.

Find the average $S_{\text{avg}}$ :

$S_{\text{avg}} = \frac{\Sigma S(i)}{n}$

Choose the cluster where $S_{\text{avg}}$ is maximum.

![](https://i.sstatic.net/iAWnF.png)

#### 3. The Calinski Harabasz Score

$\text{CH} = \frac{\text{Tr}(B_k)}{\text{Tr}(W_k)} \times \frac{n - k}{k - 1}$

Where:
- $\text{CH}$ is the Calinski-Harabasz Score.
- $\text{Tr}(B_k)$ is the trace of the **between-cluster dispersion matrix**.
- $\text{Tr}(W_k)$ is the trace of the **within-cluster dispersion matrix**.
- $n$ is the **total number of data points**.
- $k$ is the **number of clusters**.

#### **Between-Cluster Dispersion Matrix ($B_k$)**
This measures the dispersion **between different clusters**. It is defined as:

$B_k = \sum_{j=1}^{k} n_j (\mu_j - \mu)(\mu_j - \mu)^T$

Where:
- $n_j$ is the number of data points in cluster $j$ .
- $\mu_j$ is the centroid of cluster $j$ .
- $\mu$ is the overall mean of the data.

#### **Within-Cluster Dispersion Matrix ($W_k$)**
This measures the dispersion **within each cluster**. It is defined as:


$W_k = \sum_{j=1}^{k} \sum_{x_i \in C_j} (x_i - \mu_j)(x_i - \mu_j)^T$

Where:
- $x_i$ are the data points in cluster $C_j$ .
- $\mu_j$ is the centroid of cluster $j$ .


**Higher Score**: Indicates that clusters are well-separated and compact.

**Lower Score**: Suggests overlapping or poorly defined clusters.

---
---

# DBSCAN 
#### Key Parameters:
- **Epsilon (ε)**: This parameter defines the radius of the neighborhood around a point. It is the maximum distance between two points for one to be considered as in the neighborhood of the other.
- **MinPoints**: This is the minimum number of points required to form a dense region (including the point itself).

#### Types of Points:
1. **Core Points**: 
   - A Core Point is a point that has at least `MinPoints` (including itself) within its ε-radius neighborhood.
2. **Boundary Points**:
   - A Boundary Point is a point that has fewer than `MinPoints` within its ε-radius neighborhood but is has atleast one core Point in the ε-radius.
3. **Noise Points**:
   - Noise Points, also known as outliers, are points that are neither Core Points nor Boundary Points.

#### Density-connected
Density-connected means that $p$ and $q$ can be connected through a chain of Core Points where each step in the chain is within the distance ϵ.

### Algorithm
1. Identify all points as either core point, border point or noise point 
2. For all of the unclustered core points
   - Create a new cluster
   - add all the points that are unclustered and density connected to the current point into this cluster 
3. For each unclustered border point assign it to the cluster of nearest core point 
4. Leave all the noise points as it is.



---
---

# Agglomerative Hierarchical Clustering 

## Algorithm

#### 1. Initialize the Proximity Matrix
   - Calculate distances between each pair of data points (e.g., using Euclidean distance) and store them in a matrix.

#### 2. Make Each Point a Cluster
   - Treat each data point as its own cluster.

#### 3. Iterative Merging Process
   - Repeat the following steps until only one cluster remains:
     - **a. Merge the Two Closest Clusters**: Identify and merge the pair of clusters with the smallest distance.
     - **b. Update the Proximity Matrix**: Recalculate distances between the newly formed cluster and the remaining clusters based on the chosen linkage criteria (e.g., minimum, maximum, average).

#### 4. Stop When Only One Cluster Remains
   - Continue until all data points are merged into a single cluster, resulting in a hierarchy of clusters.


## Types of Linkage Criteria (Ways to Measure Distance Between Clusters)

#### 1. Single Linkage (Minimum Linkage)
   - Defines the distance between two clusters as the **minimum distance** between any point in one cluster and any point in the other.
   - **Characteristics**: Tends to create elongated, chain-like clusters; sensitive to noise or outliers.

#### 2. Complete Linkage (Maximum Linkage)
   - Defines the distance between two clusters as the **maximum distance** between any point in one cluster and any point in the other.
   - **Characteristics**: Tends to form compact, spherical clusters; less sensitive to outliers than single linkage.

#### 3. Average Linkage
   - Defines the distance between two clusters as the **average distance** between all pairs of points where one point is from one cluster and the other is from the other.
   - **Characteristics**: Compromise between single and complete linkage.

#### 4. Ward's Method
   - Minimizes the increase in **total within-cluster variance** when merging two clusters, measured as the change in error sum of squares.
   - **Characteristics**: Tends to create compact, spherical clusters; minimizes within-cluster variance at each merging step.

## Finding the Ideal Number of Clusters

1. **Plot the Dendrogram**
2. **Cut the Dendrogram Horizontally**
   - Visually inspect the dendrogram and make a horizontal cut at a certain height to define the number of clusters.

3. **Find the Longest Vertical Line**
   - Identify the longest vertical line that does not intersect with any other line, indicating the biggest distance between merged clusters and a natural division.

4. **Determine the Number of Clusters**
   - The ideal number of clusters corresponds to the number of clusters below the horizontal cut through the longest vertical line.
  
---
---
