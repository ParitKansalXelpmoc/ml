## **Content**

- [Feature Scaling](#feature-scaling)
  - [Standardization](#1-standardization)
  - [Normalization](#2-normalization)
- [Encoding Categorical Data](#encoding-categorical-data)
- [Mathematical Transformations](#mathematical-transformations)
  - [Function Transformation](#1-function-transformation)
  - [Power Transform](#2-power-transform)
  - [Quantile Transformation](#3-quantile-transformation)
- [Encoding Numerical Features](#encoding-numerical-features)
  - [Discretizationbinning](#1-discretizationbinning)
  - [Binarization](#2-binarization)
- [Outlier Handling](#outlier)
  - [Outlier Detection Techniques](#outlier-detection-techniques)
  - [Outlier Handling Techniques](#outlier-handling-techniques)
- [Handling Missing Values](#handling-missing-values)
    - [Removing](#removing)
    - [KNN Imputer](#knn-imputer)
    - [Iterative Imputer](#iterative-imputative)
- [Dimension Reduction](#dimension-reduction)
	- [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
	- [Linear Discriminant Analysis (LDA)](#linear-discriminant-analysis-lda)
 	- [Singular Value Decomposition (SVD)](#singular-value-decomposition-svd)
- [Handling Imbalanced Data](#handling-imbalanced-data)
---
---

## **Feature Scaling**
#### 1. Standardization
   - Formula: $x' = \frac{x - \text{mean}(x)}{\sigma}$
   - Standardized data has a mean of 0 and a standard deviation of 1.
   - Useful when features are on different scales; works well with algorithms that assume normal distribution.

#### 2. Normalization

- **Min-Max Scaling**: Scales data to a fixed range, typically [0, 1] or [-1, 1].

	- Formula: $x' = \frac{x - \text{min}(x)}{\text{max}(x) - \text{min}(x)}$

 	- **Used for data where negative and positive values are known.**
   
- **Max Abs Scaling**: Divides each value by the maximum absolute value in the feature, scaling between -1 and 1.

  	- Formula: $x' = \frac{x}{max(|x|)}$
       	
	- **Works well with sparse data.**

- **Robust Scaling**: Uses the Interquartile Range (IQR) instead of min and max, making it less sensitive to outliers.

  	- Formula: $x' = \frac{x - \text{median}(x)}{\text{IQR}}$
       	
	- **Works well with outliers.**
---
---

## **Encoding Categorical Data**

1. **Ordinal Encoding**
   
   ```plaintext
   Original Feature | Ordinal Feature
   -----------------|----------------
   Poor             |       0
   Good             |       1
   Excellent        |       2
   ```

2. **Nominal Encoding (One-Hot Encoding)**
   ```plaintext
   Pet Type   | Dog | Cat | Rabbit
   -----------|-----|-----|-------
   Dog        |  1  |  0  |   0
   Cat        |  0  |  1  |   0
   Rabbit     |  0  |  0  |   1
   ```
---
---

## **Mathematical Transformations**

### 1) **Function Transformation**

#### i) **Reciprocal Transformation**: $y = \frac{1}{x}$
#### ii) **Square Transformation**: $y = x^2$
   - **left-skewed data**

#### iii) **Logarithmic Transformation**: $y = \log(x + 1)$ or $y = \log(x)$
   - **right-skewed data**


#### iv) **Square Root Transformation**: $y = \sqrt{x}$

### 2) **Power Transform**

#### i) **Box-Cox Transformation**
   - Formula
	   - $x = \frac{x^\lambda - 1}{\lambda}$ for $\lambda \neq 0$
	   - $y = \log(x)$ for $\lambda = 0$
   - **Works when the data is strictly positive.**

#### ii) **Yeo-Johnson Transformation**
- Formula
	-  $y' = \frac{(y + 1)^\lambda - 1}{\lambda}, \text{ if } \lambda \neq 0, y \geq 0$
	- $y'= \log(y + 1), \text{ if } \lambda = 0, y \geq 0$
	- $y'= -\frac{(-y + 1)^{2 - \lambda} - 1}{2 - \lambda}, \text{ if } \lambda \neq 2, y < 0$
	- $y' = -\log(-y + 1), \text{ if } \lambda = 2, y < 0$
- **Works with both positive and negative values**.

### 3) **Quantile Transformation**
- **based on its cumulative distribution function (CDF)**
- #### Steps in Quantile Transformation
	1. **Sort the Data:**
	   First, the data is sorted in ascending order
	
	2. **Calculate the Rank (or Percentile) for Each Data Point:**
	   $r_i = \frac{i}{n}$
	   where $i$ is the index of the data point in the sorted list (starting from 1).
	
	3. **Map the Quantiles to a New Distribution:**
	   - **Uniform Distribution**: The simplest form of quantile transformation is to map the ranks directly to a uniform distribution over the range [0, 1]. The transformed value $y_i$ of $x_i$ is:
	     $y_i = r_i = \frac{i}{n}$
	
	   - **Normal Distribution**: If you want to transform the data into a normal distribution, you would map the quantiles $r_i$ to a corresponding value from the **inverse cumulative distribution function (CDF)** of the standard normal distribution $\Phi^{-1}(r_i)$.
  		
    		The transformed value $y_i$ becomes:

	      $y_i = \Phi^{-1}(r_i)$ where $\Phi^{-1}(r_i)$ is the inverse of the normal CDF at the quantile $r_i$. This maps the data to a normal distribution with mean 0 and variance 1.
	
	4. **Handling Ties:**
	   If there are duplicate values (ties) in the data, the rank $r_i$ is calculated by averaging the ranks of the tied values. This ensures that the transformed data remains consistent.

---
---

## **Encoding numerical features**

### 1. **Discretization/Binning**
- **Unsupervised Binning**: Bins are created without considering target variable labels.
  - **Uniform Binning (Equal Width)**
  - **Quantile Binning (Equal Frequency)**
  - **K-means Binning**
- **Supervised Binning**: Bin boundaries are created based on information from the target variable.
- **Custom Binning**: The user defines bins based on domain knowledge or data distribution.



### 2. **Binarization**
Converts continuous data into binary (0 or 1) values.

   - **Binary Encoding Rule**:
     - $y_i = 0$, if $x_i \leq \text{threshold}$
     - $y_i = 1$, if $x_i > \text{threshold}$

---
---

## **Outlier**

### Outlier Detection Techniques

1. **For Normally Distributed Data (Mean and Standard Deviation Method)**
   - $\text{Lower Bound} = \mu - 3\sigma$
   - $\text{Upper Bound} = \mu + 3\sigma$
   - Any value outside this range is considered an outlier.

3. **For Skewed Data (Interquartile Range, IQR Method)**
   - **Lower Bound**: $Q1 - 1.5 \times \text{IQR}$
   - **Upper Bound**: $Q3 + 1.5 \times \text{IQR}$
   - Any data point outside this range is considered an outlier.
   
4. **Percentile Method**
   - In this method, outliers are identified based on percentiles. Values in the lower or upper extremes (e.g., below the 1st percentile or above the 99th percentile) are marked as outliers.
   - This approach is particularly useful when dealing with highly skewed distributions or when specific cutoffs are preferred.

---

### Outlier Handling Techniques

1. **Trimming (Removing Outliers)**
   - Remove the outliers

2. **Capping**
   - Replace the outliers with a set boundary

---
---


## **Handling Missing Values**

![](https://github.com/ParitKansal/ml/blob/main/photos/MissingValues.png)

---

### Removing
Remove rows that have missing values but only if
- Check if the missing data is random.
- missing percentage is less than 5%.
- the probability density function (PDF) of numerical columns are similar before and after removal
- the distribution of categorical values are similar before and after removal

---

### KNN Imputer

| **Row** | **feature1** | **feature2** | **feature3** |
|---------|--------------|--------------|--------------|
| **0**   | 1.0          | 2.0          | 1.5          |
| **1**   | 2.0          | NaN          | 2.5          |
| **2**   | NaN          | 6.0          | 3.5          |
| **3**   | 4.0          | 8.0          | NaN          |
| **4**   | 5.0          | 10.0         | 5.5          |


#### Step 1: **Calculate Squared Euclidean Distance**

|           | **Row 0** | **Row 1** | **Row 2** | **Row 3** | **Row 4** |
|-----------|-----------|-----------|-----------|-----------|-----------|
| **Row 0** |$= (1.0 - 2.0)^2 + (1.5 - 2.5)^2 = 2$|$= (1.0 - 2.0)^2 + (1.5 - 2.5)^2= 2$|$= (2.0 - 6.0)^2 + (1.5 - 3.5)^2 = 20$|$= (1.0 - 4.0)^2 + (2.0 - 8.0)^2 = 45$|$= (1.0 - 5.0)^2 + (2.0 - 10.0)^2 + (1.5 - 5.5)^2 = 96$|
| **Row 1** | 2         |$= (2.5 - 3.5)^2 = (-1)^2 = 1$| 1         |$= (2.0 - 4.0)^2 + (2.5 - 8.0)^2= 34.25$|$= (2.0 - 5.0)^2 + (2.5 - 5.5)^2 = 18$|
| **Row 2** | 20        | 1         | 0         |$= (6.0 - 8.0)^2 = 4$|$= (6.0 - 10.0)^2 + (3.5 - 5.5)^2 = 20$|
| **Row 3** | 45        | 34.25     | 4         | 0         |$= (4.0 - 5.0)^2 + (8.0 - 10.0)^2 = 5$|
| **Row 4** | 96        | 18        | 20        | 5         | 0         |


|           | **Row 0** | **Row 1** | **Row 2** | **Row 3** | **Row 4** |
|-----------|-----------|-----------|-----------|-----------|-----------|
| **Row 0** | 0         | 2         | 20        | 45        | 96        |
| **Row 1** | 2         | 0         | 1         | 34.25     | 18        |
| **Row 2** | 20        | 1         | 0         | 4         | 20        |
| **Row 3** | 45        | 34.25     | 4         | 0         | 5         |
| **Row 4** | 96        | 18        | 20        | 5         | 0         |

#### Step 2: **Non-Euclidean Distances:**

$\sqrt{\text{Squared distace}*\frac{\text{Total No of columns}}{\text{no of cols filled in row}}}$

|           | **Row 0**                             | **Row 1**                             | **Row 2**                             | **Row 3**                             | **Row 4**                             |
|-----------|---------------------------------------|---------------------------------------|---------------------------------------|---------------------------------------|---------------------------------------|
| **Row 0** | $\sqrt{\frac{3}{3} \cdot \text{dis}(0, 0)} = 0$ | $\sqrt{\frac{3}{3} \cdot \text{dis}(0, 1)} = 1.732$ | $\sqrt{\frac{3}{3} \cdot \text{dis}(0, 2)} = 5.477$ | $\sqrt{\frac{3}{3} \cdot \text{dis}(0, 3)} = 9.486$ | $\sqrt{\frac{3}{3} \cdot \text{dis}(0, 4)} = 16.970$ |
| **Row 1** | $\sqrt{\frac{2}{3} \cdot \text{dis}(1, 0)} = 1.632$ | $\sqrt{\frac{2}{3} \cdot \text{dis}(1, 1)} = 0$ | $\sqrt{\frac{2}{3} \cdot \text{dis}(1, 2)} = 0.816$ | $\sqrt{\frac{2}{3} \cdot \text{dis}(1, 3)} = 4.268$ | $\sqrt{\frac{2}{3} \cdot \text{dis}(1, 4)} = 3.464$ |
| **Row 2** | $\sqrt{\frac{2}{3} \cdot \text{dis}(2, 0)} = 5.477$ | $\sqrt{\frac{2}{3} \cdot \text{dis}(2, 1)} = 0.816$ | $\sqrt{\frac{2}{3} \cdot \text{dis}(2, 2)} = 0$ | $\sqrt{\frac{2}{3} \cdot \text{dis}(2, 3)} = 1.632$ | $\sqrt{\frac{2}{3} \cdot \text{dis}(2, 4)} = 5.477$ |
| **Row 3** | $\sqrt{\frac{2}{3} \cdot \text{dis}(3, 0)} = 9.486$ | $\sqrt{\frac{2}{3} \cdot \text{dis}(3, 1)} = 4.268$ | $\sqrt{\frac{2}{3} \cdot \text{dis}(3, 2)} = 1.632$ | $\sqrt{\frac{2}{3} \cdot \text{dis}(3, 3)} = 0$ | $\sqrt{\frac{2}{3} \cdot \text{dis}(3, 4)} = 2.581$ |
| **Row 4** | $\sqrt{\frac{3}{3} \cdot \text{dis}(4, 0)} = 16.970$ | $\sqrt{\frac{3}{3} \cdot \text{dis}(4, 1)} = 3.464$ | $\sqrt{\frac{3}{3} \cdot \text{dis}(4, 2)} = 5.477$ | $\sqrt{\frac{3}{3} \cdot \text{dis}(4, 3)} = 2.581$ | $\sqrt{\frac{3}{3} \cdot \text{dis}(4, 4)} = 0$ |


#### Step 3: **Apply Uniform or dist method for finding missing values**
- **Uniform Method:** This method uses the average of the feature values of the K nearest neighbors. You can define the number of neighbors (K) and fill in missing values based on their average. 
- **Distance Method:** In this method, the weighted average of the feature values is calculated, where the weights are the inverse of the non-Euclidean distance: $\frac{\sum\frac{1}{dist}*\text{feature Value}}{\sum\frac{1}{dist}}$


| **Row** | **Feature 1** | **Feature 2** | **Feature 3** |
|---------|---------------|---------------|---------------|
| 0       | 1.0           | 2.0           | 1.5           |
| 1       | 2.0           |$= \frac{\left(\frac{1}{1.632} \times 2.0\right) + \left(\frac{1}{0.816} \times 6.0\right) + \left(\frac{1}{4.268} \times 8.0\right) + \left(\frac{1}{3.464} \times 10.0\right)}{\left(\frac{1}{1.632} + \frac{1}{0.816} + \frac{1}{4.268} + \frac{1}{3.464}\right)}  \approx 5.65$ | 2.5           |
| 2       |$= \frac{\left(\frac{1}{5.477} \times 1.0\right) + \left(\frac{1}{0.816} \times 2.0\right) + \left(\frac{1}{1.632} \times 4.0\right) + \left(\frac{1}{5.477} \times 5.0\right)}{\left(\frac{1}{5.477} + \frac{1}{0.816} + \frac{1}{1.632} + \frac{1}{5.477}\right)} \approx 3.82$| 6.0           | 3.5           |
| 3       | 4.0           | 8.0           |$= \frac{\left(\frac{1}{9.486} \times 1.5\right) + \left(\frac{1}{4.268} \times 2.5\right) + \left(\frac{1}{1.632} \times 3.5\right) + \left(\frac{1}{2.581} \times 5.5\right)}{\left(\frac{1}{9.486} + \frac{1}{4.268} + \frac{1}{1.632} + \frac{1}{2.581}\right)} \approx 3.75$|
| 4       | 5.0           | 10.0          | 5.5           |


---

### Iterative imputative
![Iterative imputative](https://github.com/ParitKansal/ml/blob/main/photos/Iterative%20imputative.png)

 - Initial Imputation (Fill NaN Values with Mean
 - Choose a NaN Value (X) to Impute
 - Train a model (e.g., linear regression, decision tree, etc.) where:
	- The features (inputs) are a combination of columns b and d (denoted as (b + concat d)).
	- The target variable is a combination of the values from columns a and c (denoted as concat a + c).
 - After training the model, use the input data from column e (denoted as e) to predict the missing value X.
 - Repeat for All Missing Values
 - After imputing all missing values in the dataset, update the missing values with the predicted values.
 - Instead of filling with the mean, use the newly predicted values as the starting point for the next iteration.
 - Repeat this process for several iterations or until the changes in imputed values become minimal between iterations.

---
---

# Dimension Reduction


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

---
---

## Singular Value Decomposition (SVD)

### Step-by-Step Reduction Process

1. **Perform SVD on $\mathbf{A}$**:
   - Decompose $\mathbf{A}$ as:
     $\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$
     where:
     - $\mathbf{U}$ is an $m \times m$ matrix of the left singular vectors.
     - $\mathbf{\Sigma}$ is an $m \times n$ diagonal matrix containing the singular values in descending order.
     - $\mathbf{V}^T$ is the transpose of an $n \times n$ matrix $\mathbf{V}$, which contains the right singular vectors.

2. **Truncate $\mathbf{U}$, $\mathbf{\Sigma}$, and $\mathbf{V}^T$**:
   - Since the goal is to reduce $\mathbf{A}$ to an $m \times 2$ matrix, we keep only the top 2 singular values and the corresponding left and right singular vectors.
   - Define:
     - $\mathbf{U}_2$: the first 2 columns of $\mathbf{U}$, with dimensions $m \times 2$.
     - $\mathbf{\Sigma}_2$: a $2 \times 2$ diagonal matrix containing the top 2 singular values.
     - $\mathbf{V}_2^T$: the first 2 rows of $\mathbf{V}^T$, with dimensions $2 \times n$.

   Then, we can approximate $\mathbf{A}$ as:
   $\mathbf{A} \approx \mathbf{U}_2 \mathbf{\Sigma}_2 \mathbf{V}_2^T$

3. **Compute the Reduced Representation $\mathbf{A}'$**:
   - To get the reduced matrix $\mathbf{A}'$ of dimensions $m \times 2$, multiply $\mathbf{U}_2$ and $\mathbf{\Sigma}_2$:
   $\mathbf{A}' = \mathbf{U}_2 \mathbf{\Sigma}_2$
   This yields an $m \times 2$ matrix $\mathbf{A}'$, which is the desired reduced representation.

### SVD Calculation Steps (Detailed)

The following steps outline the SVD calculation in detail, which are foundational for deriving $\mathbf{U}$, $\mathbf{V}$, and $\mathbf{\Sigma}$:

1. **Calculate Eigenvalues and Eigenvectors of $A \cdot A^T$**:
   - Compute $A \cdot A^T$.
   - Find the eigenvalues and eigenvectors of $A \cdot A^T$.
   - These eigenvectors form the columns of $\mathbf{U}$.

2. **Normalize Each Eigenvector**:
   - Normalize each eigenvector of $A \cdot A^T$ to ensure the columns of $\mathbf{U}$ have unit length.

3. **Stack Eigenvectors of $A \cdot A^T$ Based on Eigenvalues (Descending)**:
   - Sort the eigenvalues in descending order.
   - Stack the corresponding normalized eigenvectors horizontally to form $\mathbf{U}$.

4. **Calculate Eigenvalues and Eigenvectors of $A^T \cdot A$**:
   - Compute $A^T \cdot A$.
   - Find the eigenvalues and eigenvectors of $A^T \cdot A$.
   - These eigenvectors form the columns of $\mathbf{V}$.

5. **Normalize Each Eigenvector**:
   - Normalize each eigenvector of $A^T \cdot A$ so the columns of $\mathbf{V}$ have unit length.

6. **Stack Eigenvectors of $A^T \cdot A$ Based on Eigenvalues (Descending)**:
   - Sort the eigenvalues in descending order.
   - Stack the corresponding normalized eigenvectors horizontally to form $\mathbf{V}$.

7. **Transpose $V$**:
   - Compute $V^T$, which is used in the SVD decomposition.

8. **Form $\Sigma$**:
   - Use a zero matrix of dimensions $m \times n$.
   - Populate this matrix with the square roots of the eigenvalues of $A^T \cdot A$ along the diagonal, in descending order. These values are the singular values of $A$.

This process yields the matrices $\mathbf{U}$, $\mathbf{\Sigma}$, and $\mathbf{V}$ required for the SVD of $\mathbf{A}$ and enables us to obtain the reduced form $\mathbf{A}'$ as described.

---
---


## **Handling Imbalanced Data**

### 1. **Undersampling and Oversampling**  
![](https://github.com/ParitKansal/ml/blob/main/photos/1_7xf9e1EaoK5n05izIFBouA.webp)

### 2. **SMOTE (Synthetic Minority Oversampling Technique)**  
- **Train a k-NN model** on minority class observations:
  - Identify the **k nearest neighbors** for each minority class sample (commonly \( k = 5 \)).  
- **Create synthetic data**:
  1. **Select 1 example** randomly from the minority class.  
  2. **Select one neighbor** randomly from its $k$-nearest neighbors.  
  3. Extract a **random number $\alpha$** between 0 and 1 for interpolation.  
  4. Generate the synthetic sample using the formula:  $\text{Synthetic sample} = \text{Original sample} + \alpha \times (\text{Neighbor} - \text{Original sample})$
- Repeat the process to create multiple synthetic samples.
- **Combine the original dataset with synthetic samples** to form a balanced dataset.

### 3. Ensemble Methods
  ![](https://github.com/ParitKansal/ml/blob/main/photos/tempo231412.png)

### 4. Giving Weights to Different Classes  


---
---

