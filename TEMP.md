---
---
## Principal Component Analysis (PCA)
- The aim of Principal Component Analysis (PCA) is to find a direction or vector onto which the projection of data points will have the maximum variance. This direction is called the principal component. PCA identifies the directions (principal components) in which the data varies the most and projects the data onto these directions to reduce dimensionality while retaining the most significant information.

### Steps for Principal Component Analysis (PCA)

1. **Mean Centering the Data**:  
   - Subtract the mean of each feature from the dataset, making the data zero-centered (mean-centered).

2. **Compute the Covariance Matrix**:
   - For an $n \times m$ matrix $\text{Data}$, where $n$ is the number of observations (e.g., 1000) and $m$ is the number of features (e.g., 4), calculate the covariance matrix. 
   - The covariance matrix, $C$, is an $m \times m$ (e.g., $4 \times 4$) matrix where each element represents the covariance between two features:

		|          |      $f_1$              |     $f_2$              | $\dots$            | $f_m$                  |
		|----------|-------------------------|------------------------|--------------------|------------------------|
		| $f_1$    | $\text{cov}(f_1, f_1)$  | $\text{cov}(f_1, f_2)$ | $\dots$            | $\text{cov}(f_1, f_m)$ |
		| $f_2$    | $\text{cov}(f_2, f_1)$  | $\text{cov}(f_2, f_2)$ | $\dots$            | $\text{cov}(f_2, f_m)$ |
		| $\vdots$ | $\vdots$                | $\vdots$               | $\ddots$           | $\vdots$               |
		| $f_m$    | $\text{cov}(f_m, f_1)$  | $\text{cov}(f_m, f_2)$ | $\dots$            | $\text{cov}(f_m, f_m)$ |
   - Diagonal elements represent variances, and off-diagonal elements represent covariances between different features.

3. **Calculate Eigenvalues and Eigenvectors**:
   - Solve for the eigenvalues and eigenvectors of the covariance matrix $C$. The eigenvalues indicate the amount of variance captured by each principal component.
   - The equation for finding eigenvalues $\lambda$ and eigenvectors $X$ is:

     $(C - \lambda I)X = 0$

     where $I$ is the identity matrix.

4. **Select Principal Components**:
   - Sort the eigenvalues in descending order, and select the top p(e.g., 2) eigenvalues along with their corresponding eigenvectors. These p eigenvectors form a new $p \times m$ eigenvector matrix (e.g., $2 \times 4$) that will be used for projection.

5. **Project Data onto New Subspace**:
   - Transform the original data to a lower-dimensional space by multiplying it with the selected eigenvectors:

     $Data_{new} = Data_{MeanCentered} . Eigenvectors$
     
   - If you reduce from m features to p (e.g., 4 features to 2), the result will be a $n \times p$ matrix (e.g., $1000 \times 2$ matrix).


---
---

## Linear Discriminant Analysis (LDA)

Given a dataset with $c$ classes, let:
- $\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n]$ represent the dataset with $n$ samples.
- Each sample $\mathbf{x}_i \in \mathbb{R}^d$ is a $d$-dimensional feature vector.
- Let $\mathbf{X}_j$ be the subset of samples belonging to class $j$, which has $n_j$ samples.

1. **Class Mean Vector**: $m_j = \frac{1}{n_j} \sum_{x \in X_j} x$
2. **Within-Class Scatter Matrix**: The spread of each class around its own mean. $S_W = \sum_{j=1}^{c} \sum_{x \in X_j} (x - m_j)(x - m_j)^T$
3. **Between-Class Scatter Matrix**: How far apart the class means are from the overall mean. $S_B = \sum_{j=1}^{c} n_j (m_j - m)(m_j - m)^T$
4. **Optimization Objective**: find a projection vector w that maximizes the ratio of between-class variance to within-class variance. $J(w) = \frac{w^T S_B w}{w^T S_W w}$
5. **Eigenvalue Problem**: $S_B w = \lambda S_W w$
	1. Compute the eigenvalues and eigenvectors of $\mathbf{S}_W^{-1} \mathbf{S}_B$.
	2. Sort the eigenvectors by their corresponding eigenvalues in descending order.
	3. Select the top $k$ eigenvectors (where $k \leq c - 1$) to form the transformation matrix $\mathbf{W}$.

### Imp Points
- **Assumptions:** LDA assumes that each class follows a Gaussian distribution with identical covariance matrices, and it is sensitive to outliers and class imbalances.
- Aims to maximize the distace b/w classes and minimise the variance within each class' data points
- **Computationally Efficient:** LDA is faster than some non-linear techniques (e.g., t-SNE or UMAP) because it is a linear method.
- **LDA vs. PCA**
	- PCA: An unsupervised method that focuses on maximizing variance, capturing the most important features regardless of class labels.
	- LDA: A supervised method that maximizes class separability, explicitly using class labels to create dimensions that emphasize distinctions between classes.

