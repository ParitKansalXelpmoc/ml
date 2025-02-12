
## Content
- **Data Preprocessing**
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
- **Metrics**
    - [Classification Metrics](#classification-metrics)
    - [Regression Metrics](#regression-metrics)
- **ML Models**
    - [**Linear Regression**](#linear-regression)
        - [Polynomial Regression](#polynomial-regression)
    - [**Ridge, Lasso, Elastic Regression**](#ridge-lasso-elastic-regression)
    - [**Logistic regression**](#logistic-regression)
        - Solutions
            - [Percepton Trick](#1-percepton-trick)
            - [Sigmoid function](#2-sigmoid-function)
            - [Maximum Likelihood](#3-maximum-likelihood)
        - [**Softmax Regression / Multinomial Logistic Regression**](#softmax-regression--multinomial-logistic-regression)
    - [**KNN**](#knn)
    - [**Naive Bayes Classifier**](#naive-bayes-classifier)
    - [Handling Numerical Values](#handling-numerical-values)
    - [**CART**](#cart---classification-and-regression-trees)
    - [**Decision Tree**](#decision-tree)
    - [**Regression Trees**](#regression-trees)
    - [Feature Importance Using Decision Tree](#feature-importance-for-decision-tree-like-algos)
    - [**SVM**](#svm-support-vector-machine)
    - [Ensemble Learning](#ensemble-learning)
        - [**Voting Ensemble**](#voting)
        - [**Stacking**](#stacking)
        - [**Bagging**](#bagging-techniques)
        - Boosting
            - [**Ada Boosting**](#ada-boosting)
            - [**Gradient Boosting**](#gradient-boosting)
                - [For Regression](#gradient-boosting-for-regression)
                - [For Classification](#gradient-boosting-for-classification)
                - [For Every Loss Fun](#algorithmapplicale-for-every-loss-fun)
            - [**XGBoost**](#xgboost)
                - [Why XGBoost](#why-xgboost)
                - [For Regression](#xgboost-for-regression)
                - [For Classification](#xgboost-for-classification)
                - [Mathmatics For Xgboost](#mathmatics-for-xgboost)
            - [Bagging Vs Random Forest](#bagging-vs-random-forest)
    - Clustering
        - [**K-Means**](#k-means)
        - [**DBSCAN**](#dbscan)
        - [**Agglomerative Hierarchical Clustering**](#agglomerative-hierarchical-clustering)
- **NLP**
    - [Text Preprocessing Steps](#text-preprocessing-steps)
    - [Text Representation & Text Embeddings](#text-representation--text-embeddings)
        - [One-Hot Encoding](#one-hot-encoding)
        - [Bag of Words (BoW)](#bag-of-wordbow)
        - [N-Gram](#n-gram)
        - [TF-IDF (Term Frequency-Inverse Document Frequency)](#tf-idf-term-frequency-inverse-document-frequency)
        - [Word2Vec](#word2vec)

------

# **Data Preprocessing**


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

#### **Step 1: Calculate Squared Euclidean Distance**
We compute the squared Euclidean distance for each pair of rows using only the available (non-missing) values.

$\text{Distance}(A, B) = \sum (\text{feature}_i^{(A)} - \text{feature}_i^{(B)})^2$

|           | **Row 0** | **Row 1** | **Row 2** | **Row 3** | **Row 4** |
|-----------|-----------|-----------|-----------|-----------|-----------|
| **Row 0** | 0         | 2         | 20        | 45        | 96        |
| **Row 1** | 2         | 0         | 1         | 34.25     | 18        |
| **Row 2** | 20        | 1         | 0         | 4         | 20        |
| **Row 3** | 45        | 34.25     | 4         | 0         | 5         |
| **Row 4** | 96        | 18        | 20        | 5         | 0         |

#### **Step 2: Adjust Distance Using Missing Values**
We modify the squared Euclidean distance to account for missing values using the formula:

$\text{Adjusted Distance} = \sqrt{\text{Squared Distance} \times \frac{\text{Total Columns}}{\text{Columns Used}}}$

where:
- **Total Columns** = 3 (feature1, feature2, feature3)
- **Columns Used** = Number of non-missing values used in distance computation.

|           | **Row 0** | **Row 1** | **Row 2** | **Row 3** | **Row 4** |
|-----------|-----------|-----------|-----------|-----------|-----------|
| **Row 0** | $\sqrt{0 \times \frac{3}{3}} = 0.000$ | $\sqrt{2 \times \frac{3}{2}} = \sqrt{3} = 1.732$ | $\sqrt{20 \times \frac{3}{2}} = \sqrt{30} = 5.477$ | $\sqrt{45 \times \frac{3}{2}} = \sqrt{67.5} = 9.486$ | $\sqrt{96 \times \frac{3}{3}} = \sqrt{96} = 16.970$ |
| **Row 1** | $\sqrt{2 \times \frac{3}{2}} = \sqrt{3} = 1.732$ | $\sqrt{0 \times \frac{3}{3}} = 0.000$ | $\sqrt{1 \times \frac{3}{1}} = \sqrt{3} = 0.816$ | $\sqrt{34.25 \times \frac{3}{1}} = \sqrt{102.75} = 4.268$ | $\sqrt{18 \times \frac{3}{2}} = \sqrt{27} = 3.464$ |
| **Row 2** | $\sqrt{20 \times \frac{3}{2}} = \sqrt{30} = 5.477$ | $\sqrt{1 \times \frac{3}{1}} = \sqrt{3} = 0.816$ | $\sqrt{0 \times \frac{3}{3}} = 0.000$ | $\sqrt{4 \times \frac{3}{2}} = \sqrt{6} = 2.000$ | $\sqrt{20 \times \frac{3}{2}} = \sqrt{30} = 4.472$ |
| **Row 3** | $\sqrt{45 \times \frac{3}{2}} = \sqrt{67.5} = 9.486$ | $\sqrt{34.25 \times \frac{3}{1}} = \sqrt{102.75} = 4.268$ | $\sqrt{4 \times \frac{3}{2}} = \sqrt{6} = 2.000$ | $\sqrt{0 \times \frac{3}{3}} = 0.000$ | $\sqrt{5 \times \frac{3}{2}} = \sqrt{7.5} = 2.236$ |
| **Row 4** | $\sqrt{96 \times \frac{3}{3}} = \sqrt{96} = 16.970$ | $\sqrt{18 \times \frac{3}{2}} = \sqrt{27} = 3.464$ | $\sqrt{20 \times \frac{3}{2}} = \sqrt{30} = 4.472$ | $\sqrt{5 \times \frac{3}{2}} = \sqrt{7.5} = 2.236$ | $\sqrt{0 \times \frac{3}{3}} = 0.000$ |

|           | **Row 0** | **Row 1** | **Row 2** | **Row 3** | **Row 4** |
|-----------|-----------|-----------|-----------|-----------|-----------|
| **Row 0** | 0.000     | 1.732     | 5.477     | 9.486     | 16.970    |
| **Row 1** | 1.732     | 0.000     | 0.816     | 4.268     | 3.464     |
| **Row 2** | 5.477     | 0.816     | 0.000     | 2.000     | 4.472     |
| **Row 3** | 9.486     | 4.268     | 2.000     | 0.000     | 2.236     |
| **Row 4** | 16.970    | 3.464     | 4.472     | 2.236     | 0.000     |

#### Step 3: **Apply Uniform or dist method for finding missing values**
- **Uniform Method:** This method uses the average of the feature values of the K nearest neighbors. You can define the number of neighbors (K) and fill in missing values based on their average. 
- **Distance Method:** In this method, the weighted average of the feature values is calculated, where the weights are the inverse of the non-Euclidean distance: $\frac{\sum\frac{1}{dist}*\text{feature Value}}{\sum\frac{1}{dist}}$

| **Row** | **feature1** | **feature2** | **feature3** |
|---------|-------------|-------------|-------------|
| **0**   | 1.0         | 2.0         | 1.5         |
| **1**   | 2.0         | **$\frac{\left(\frac{1}{1.732} \times 2.0\right) + \left(\frac{1}{3.464} \times 6.0\right) + \left(\frac{1}{4.268} \times 8.0\right) + \left(\frac{1}{5.000} \times 10.0\right)}{\frac{1}{1.732} + \frac{1}{3.464} + \frac{1}{4.268} + \frac{1}{5.000}}$= $\frac{\left(1.154 \times 2\right) + \left(0.289 \times 6\right) + \left(0.234 \times 8\right) + \left(0.200 \times 10\right)}{1.154 + 0.289 + 0.234 + 0.200}$= $\frac{2.308 + 1.732 + 1.872 + 2.000}{1.877} = \mathbf{5.71}$** | 2.5         |
| **2**   | **$\frac{\left(\frac{1}{5.477} \times 1.0\right) + \left(\frac{1}{1.225} \times 2.0\right) + \left(\frac{1}{2.000} \times 4.0\right) + \left(\frac{1}{4.472} \times 5.0\right)}{\frac{1}{5.477} + \frac{1}{1.225} + \frac{1}{2.000} + \frac{1}{4.472}}$= $\frac{\left(0.183 \times 1\right) + \left(0.816 \times 2\right) + \left(0.500 \times 4\right) + \left(0.224 \times 5\right)}{0.183 + 0.816 + 0.500 + 0.224}$= $\frac{0.183 + 1.632 + 2.000 + 1.120}{1.723} = \mathbf{2.70}$** | 6.0         | 3.5         |
| **3**   | 4.0         | 8.0         | **$\frac{\left(\frac{1}{9.486} \times 1.5\right) + \left(\frac{1}{4.268} \times 2.5\right) + \left(\frac{1}{2.000} \times 3.5\right) + \left(\frac{1}{1.000} \times 5.5\right)}{\frac{1}{9.486} + \frac{1}{4.268} + \frac{1}{2.000} + \frac{1}{1.000}}$= $\frac{\left(0.105 \times 1.5\right) + \left(0.234 \times 2.5\right) + \left(0.500 \times 3.5\right) + \left(1.000 \times 5.5\right)}{0.105 + 0.234 + 0.500 + 1.000}$= $\frac{0.158 + 0.585 + 1.750 + 5.500}{1.839} = \mathbf{3.85}$** |
| **4**   | 5.0         | 10.0        | 5.5         |

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



---

## Metrics

### Classification Metrics

| **Metric**      | **Formula**                                                                             | **Description**                             |
|-----------------|-----------------------------------------------------------------------------------------|---------------------------------------------|
| **Accuracy**    | $\frac{TP + TN}{TP + TN + FP + FN}$                                                     | Greater values indicate better performance. |
| **Precision**   | $\frac{TP}{TP + FP} = \frac{\text{True Positive}}{\text{Predicted Positive}}$           | Greater values indicate better performance. |
| **Recall**      | $\frac{TP}{TP + FN} = \frac{\text{True Positive}}{\text{Real Positive}}$                | Greater values indicate better performance. |
| **F1-score**    | $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$                             | Greater values indicate better performance. |
| **Log Loss**    | $- \frac{1}{n} \sum \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$ | Lower values indicate better performance.   |

---

### Regression Metrics

| **Metric** | **Formula** | **Description** |
|------------|-------------|-----------------|
| **Mean Absolute Error (MAE)**      | $\frac{1}{n} \sum \|y_i - \hat{y}_i\|$                    | |
| **Mean Squared Error (MSE)**       | $\frac{1}{n} \sum (y_i - \hat{y}_i)^2$                    | |
| **Root Mean Squared Error (RMSE)** | $\sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}$             | |
| **R-squared (R²)**                 | $1 - \frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y}_i)^2}$ | Greater values indicate better performance. |
| | | $0 ≤ \sum(y_i-\hat{y}_i)^2 ≤ \sum(y_i-\bar{y}_i)^2$ |
| | | $⇒ 0 ≤ R^2 ≤ 1 $ |
| **Adjusted R-squared $R^2_{adj}$**             |$1 - \frac{(1 - R^2)(N - 1)}{N - p - 1}$ | useful for comparing models with different feature sets. |

---

---
---

# **ML Models**

---

# Linear Regression
   
   - $X$: The matrix of input features (with dimensions $1000 \times 10$, where $1000$ is the number of observations and $9$ is the number of predictors and first column containing 1 only).
   - $Y$: The vector of observed outcomes (with dimensions $1000 \times 1$).
   - $\beta$: The vector of estimated coefficients (with dimensions $10 \times 1$).
   - $\hat{Y}$: The vector of predicted values (with dimensions $1000 \times 1$).

**Predicting Values ($\hat{Y}$)**:
 $\hat{Y} = X\beta$

**1. Closed Form Formula / The Ordinary Least Squares (OLS)**

 $\beta = (X^T X)^{-1} X^T Y$

**2. Non-Closed Form Formula**
Run these for n epoches

$\frac{dL}{d\beta} = -2X^T Y + 2X^T X \beta$ 

$\beta_{n} = \beta_{n-1} - \frac{\alpha}{1000} \frac{dL}{d\beta}$ .

---
---

## Polynomial Regression

Suppose we have three features and we want apply degree 2 polynonial features then calculate or make ney features -> $x^2 , y^2, z^2, xy, xz, yz$. Now apply normal linear regression.

---
---
# Ridge, Lasso, Elastic Regression

### **Ridge Regression**

$L = \sum_{i=0}^{m} (y - \hat{y})^2 + \lambda \sum \beta^2 $

$\frac{\partial L}{\partial \beta} = 2 X^T X \beta - 2 X^T Y + 2 \lambda I \beta$

- Here $I = I$ where $I[0][0] = 0$
  - Example
		<table>
		    <tr>
		        <td>0</td>
		        <td>0</td>
		        <td>0</td>
		    </tr>
		    <tr>
		        <td>0</td>
		        <td>1</td>
		        <td>0</td>
		    </tr>
		    <tr>
		        <td>0</td>
		        <td>0</td>
		        <td>1</td>
		    </tr>
		</table>

$\beta = (X^T X + \lambda I)^{-1} X^T Y$

$\beta = \beta_0 - \alpha \frac{\partial L}{\partial \beta_0}$

$m = \frac{\sum (y_i - \bar{y})(x_i - \bar{x})}{\sum (x_i - \bar{x})^2 + \lambda} \rightarrow \neq 0$

- **As** $\lambda$ **increases, value only decreases but never goes to 0**


### **Lasso Regression**

$\text{Loss, } L = \sum (y - \hat{y})^2 + \lambda ||w||$

- For $m > 0$

  $m = \frac{\sum (y_i - \bar{y})(x_i - \bar{x}) - \lambda}{\sum (x_i - \bar{x})^2}$

- For $m = 0$

  $m = \frac{\sum (y_i - \bar{y})(x_i - \bar{x})}{\sum (x_i - \bar{x})^2} $

- For $m < 0$

  $m = \frac{\sum (y_i - \bar{y})(x_i - \bar{x}) + \lambda}{\sum (x_i - \bar{x})^2}$

- **Lasso regression is used for feature selection, for greater values of** $\lambda$.

### **Properties of Ridge and Lasso Regression**

1) **How coefficients get affected as** $\lambda$ **increases**
   - For **Ridge** → $m \approx 0$ but $m \neq 0$
   - For **Lasso** → $m = 0$

2) **More the value of** $m$ **the higher & more it decreases fully as** $\lambda$ **increases.**

   ![](https://github.com/ParitKansal/ml/blob/main/photos/Untitled%20(3).png)

3) **Bias-Variance tradeoff**

![](https://github.com/ParitKansal/ml/blob/main/photos/bias%20and%20variance%20curves.webp)

4) **Impact on loss function**
   - As $\lambda$ increases, minimum loss increases.
   - As $\lambda$ increases, $m$ gets close to 0.
![](https://github.com/ParitKansal/ml/blob/main/photos/1_EyLt1w1eXHVELljki2hYuQ%20(1).png)

### **Elastic Regression**

Loss,  L = &sum;<sub>i=1</sub> (&#770;y<sub>i</sub> - y<sub>i</sub>)<sup>2</sup> + &alpha; &sum;<sub>i=1</sub> |w<sub>i</sub>| + &beta; &sum;<sub>i=1</sub> w<sub>i</sub><sup>2</sup>

- **Use when we do not know if we have to apply Ridge or Lasso regression.**
- **Ridge** → when every independent variable has some importance.
- **Lasso** → when we want to train the model on a subset of variables instead of all variables.

---
---

# Logistic regression

### 1. Percepton Trick
- If a point is in wrong region then move line towards the point
- Do not work bcoz it stops once a boundry is created thus not give best result
- ALGORITHM
   - for n in range(1000):
      - select a random col i 
         - $W_{new} = W_{old} + \eta(Y_i - \hat{Y}_i)x_i^T$
      - $\hat{Y} = X W$
   - **$W$**: The updated weight vector of dimension 10×1.
   - **$\eta$**: The learning rate.
   - **$x_i$**: The feature vector corresponding to the $i$ th data point of dimension 1×10.

### 2. Sigmoid function

![](https://mathworld.wolfram.com/images/eps-svg/SigmoidFunction_701.svg)
- Here we use sigmoid function
- ALGORITHM
   - for n in range(1000):
      - select a random col i 
         - $W_{new} = W_{old} + \eta(Y_i - \hat{Y}_i)x_i^T$
      - $\hat{Y} = \sigma(X W)$
  
### 3. Maximum Likelihood
$\text{Loss} = -\frac{1}{m} \sum_{i=1}^{m} \left( y_i \cdot \log(\hat{y}_i) + (1 - y_i) \cdot \log(1 - \hat{y}_i) \right)$

$\text{Loss} = Y\cdot\log(\hat{Y}) + (1 - Y)\cdot\log(1 - \hat{Y}) $

$W_{new} = W_{new} - \eta \frac{dL}{dW}$ .

$\hat{Y} = \sigma(X W)$

- ALGORITHM
   - For epoch in range(10):
      - $W = W_{n-1} + \frac{\eta}{m} X^T (Y - \hat{Y})$
      - $\hat{Y} = \sigma(X W)$

---
---

## Softmax Regression / Multinomial Logistic Regression
**1. Method-One**
- Apply one hot encoding

 ![](https://i.ibb.co/bW3qrVm/Untitled-1.png)
- Train model for each column

**2. Method-Two**

$\text{Loss} = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{k} \left( y^{(i)}_k \cdot \log(\hat{y}^{(i)}_k) \right)$

## Imp Points
- We can apply L1, L2, elatic net regression

---
---
# KNN
It based on the concept "You are the average of the five people you spend the most time with"

### Steps Involved:
1. Normalize the Data
2. Find the Distance of All Points:    
     - Use Euclidean distance
3. Identify the K Nearest Neighbors
4. Determine the Output:
   - KNN for Classification:
     - **Majority Vote**: The label of the query point is determined by the majority class among the K nearest neighbors.
   - KNN for Regression:
     - **Average (or Weighted Average)**: The predicted value for the query point is the average (or a weighted average) of the values of the K nearest neighbors.
---
---
# Naive Bayes Classifier

$P(A|B) = \frac{P(A ∩ B)}{P(B)}$

$P(A|B) = \frac{P(B|A)\cdot P(A)}{P(B)}$

$P\left(\frac{Won}{A∩B∩C}\right) = \frac{P\left(\frac{A∩B∩C}{Won}\right) \cdot P(Won)}{P(A∩B∩C)}$

$P\left(\frac{A∩B∩C}{Won}\right) = P\left(\frac{A,B,C}{Won}\right) = P\left(\frac{A}{Won}\right) \cdot P\left(\frac{B}{Won}\right) \cdot P\left(\frac{C}{Won}\right)$

$P\left(\frac{Won}{A∩B∩C}\right) = \frac{P\left(\frac{A}{Won}\right) \cdot P\left(\frac{B}{Won}\right) \cdot P\left(\frac{C}{Won}\right) \cdot P(Won)}{P(A∩B∩C)}$

$P\left(\frac{Won}{A∩B∩C}\right) ∝ {P\left(\frac{A}{Won}\right) \cdot P\left(\frac{B}{Won}\right) \cdot P\left(\frac{C}{Won}\right) \cdot P(Won)}$

$P\left(\frac{Loss}{A∩B∩C}\right) ∝ {P\left(\frac{A}{Loss}\right) \cdot P\left(\frac{B}{Loss}\right) \cdot P\left(\frac{C}{Loss}\right) \cdot P(Loss)}$


### Example
| Toss  | Venue    | Outlook   | Result |
|-------|----------|-----------|--------|
| Won   | Mumbai   | Overcast  | Won    |
| Lost  | Chennai  | Sunny     | Won    |
| Won   | Kolkata  | Sunny     | Won    |
| Won   | Chennai  | Sunny     | Won    |
| Lost  | Mumbai   | Sunny     | Lost   |
| Won   | Mumbai   | Sunny     | Lost   |
| Lost  | Chennai  | Overcast  | Lost   |
| Won   | Kolkata  | Overcast  | Lost   |
| Won   | Mumbai   | Sunny     | Won    |

$P\left( \frac{Won}{Lost, Mumbai, Sunny} \right) ∝ {P\left(\frac{Lost}{Won}\right) \cdot P\left(\frac{Mumbai}{Won}\right) \cdot P\left(\frac{Sunny}{Won}\right) \cdot P(Won)} = 
\frac{1}{5} \cdot \frac{2}{5} \cdot \frac{4}{5} \cdot \frac{5}{9}$

### Handling Numerical values

For numerical features, the Naive Bayes classifier uses the Gaussian (normal) distribution:

$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}$


You would apply this formula to compute the probability density for numerical values in each class, such as "Won" or "Loss" by finding the std and mean for that class and putting those and value of x in the formula to get probablity.

---
---

# CART - Classification and Regression Trees

**Pseudo code**
1. Begin with your training dataset, which should have some feature variables and classification or regression output.
2. Determine the “best feature” in the dataset to split the data on
3. Split the data into subsets that contain the correct values for this best feature. This splitting basically defines a node on the tree i.e each node is a splitting point based on a certain feature from our data.
4. Recursively generate new tree nodes by using the subset of data created from step 3.

**Advantages** 
- The cost of using the tree for inference is logarithmic in the number of data points used to train the tree

**Disadvantages** 
- Overfitting
- Prone to errors for imbalanced datasets

## Decision Tree

### Entropy

$\text{E}(x) = -\sum_{i=1}^{c} p_i\log_2(p_i)$

where $p_i$ is simply the frequentist probability of an element class $i$ in our data.

![Untitled-2-1.png](https://i.postimg.cc/cHtpvxn1/Untitled-2-1.png)

**Observation**
- For a 2 class problem the min entropy is 0 and the max is 1
- For more than 2 classes the min entropy is 0 but the max can be greater than 1
- Both $log_2$ or $log_e$ can be used to calculate entropy

### Gini impurity

$G = 1 - \sum P_i^2$

**Some times Gini Impurity may give balanced tree incomparision to entropy**

[![Untitled1-1.png](https://i.postimg.cc/YSwTLMRQ/Untitled1-1.png)](https://postimg.cc/18JvLxLz)


### Information Gain 

$\text{Information Gain} = \text{Entropy}(Parent) - \frac{1}{\text{Total Weight}}\sum Weight_i*\text{Entropy}(child_i)$

$\text{Information Gain} = \text{Gini Impurity}(Parent) - \frac{1}{\text{Total Weight}}\sum Weight_i*\text{Gini Impurity}(child_i)$

### ALGORITHM
- Calculate Entropy / Gini impurity of Parent
- Calculate Information Gain for all the columns
   - Calculate Entropy / Gini impurity for Children
   - Calculate weighted Entropy / Gini impurity of Children
- Whichever column has the highest Information Gain(maximum decrease in entropy) the algorithm will select that column to split the data.
- Once a leaf node is reached ( Entropy = 0 ), no more splitting is done.

### Calculate Information Gain for Numerical data column
- Sort the data on the basis of numerical column
- Split the entire data on the basis of every value of numerical column
- Calculate Information Gain for each split
- Select Max Information Gain and it is the Information Gain for that column

[![Untitled.png](https://i.postimg.cc/k5Zp9H7H/Untitled.png)](https://postimg.cc/WDGnwXFm)


## Regression Trees
### ALGORITHM
- Calculate Standard Deviation of the Parent Node
- Calculate Information Gain for all the columns
   - Calculate Standard Deviation for Child Nodes
      - For Categorical Features: Split the data into different groups based on the unique labels of the categorical feature. Compute the standard deviation of the target variable for each group (child node).
      - For Numerical Features: Perform repeated splits for each unique value of the numerical feature. For each possible split, divide the data into two groups (child nodes) and calculate the standard deviation of the target variable within each group.
   - Calculate Weighted Standard Deviation of Children
   - Calculate Information Gain: Information Gain = Std of Parent − Weighted Std of Children
- Whichever column has the highest Information Gain the algorithm will select that column to split the data.
- The algorithm recursively repeats the process for each child node until a stopping criterion is met.
- At each leaf node, the output is the mean of the target variable values within that node..

---
---

# Feature importance for decision tree like algos
Calculate this for each node t and which has split for feature $i$

$Feature Importance(i) = \sum_{t \in \text{nodes where feature } i \text{ is used}} \frac{N_t}{N} \left( impurity - \frac{N_{t_r}}{N_t} \cdot RightImpurity - \frac{N_{t_l}}{N_t} \cdot LeftImpurity \right)$

- $Feature Importance(i)$:  Importance score for feature $i$.
- $N_t$: Number of samples at node $t$.
- $N$: Total number of samples.
- $\text{impurity}$: Impurity measure at node $t$ (e.g., Gini impurity, entropy).
- $N_{t_r}$: Number of samples in the right child node after the split.
- $N_{t_l}$: Number of samples in the left child node after the split.
- $\text{Right Impurity}$: Impurity of the right child node of node t.
- $\text{Left Impurity}$: Impurity of the left child node of node t.

---
---
# SVM (Support Vector Machine)
![](https://i.postimg.cc/N0YydtDp/Untitled.png)

- The aim of this is to 
    - maximise the distance between $π_+$ and $π_-$
    - minimise the distance distance of wrong output points to there repective **correct margin plane**

Below objective function seeks to minimize a combination of the margin (through the regularization term) and the misclassification error (through the slack variables). The goal is to find the optimal $w*$ and $b*$ that achieve this balance.

- $\arg\min_{w*, b*} \left( Margin Error + Classification Error\right)$
- $\arg\min_{w*, b*} \left( \frac{1}{2} ||w||^2 + C \sum_{i=1}^n \zeta_i \right)$

**where**:

**$w$** : Weight vector.

**$b$** : Bias term.

**$C $** : Regularization parameter.

**$\frac{1}{2} ||w||^2 $** : Regularization term that penalizes large weights.

**$\sum_{i=1}^n \zeta_i $** : Sum of the slack variables $𝜁_𝑖$​, which represent the amount by which each data point deviates from the correct classification margin.


**We use Kernels for non linear seperable data**

---
---

# Ensemble Learning
Based on the concept of the wisdom of the crowd, decisions made by multiple models have a higher chance of being correct. Ensemble learning helps convert a low-bias, high-variance model into a low-bias, low-variance model because the random/outliers are distributed among various models instead of going to a single model.

- **Voting Ensemble**: In this approach, different models of various types, such as SVM, Decision Tree, and Logistic Regression, calculate results. For classification tasks, the output with the highest frequency is selected as the final result, or we add probabilities of each class obtained from models. For regression tasks, the mean of the outputs is calculated.

- **Stacking**: In stacking, different models of various types, such as SVM, Decision Tree, and Logistic Regression, are used to calculate results. The outputs of these models are then used to train a final model, such as KNN, to obtain the final output. This final model effectively assigns a weight to each previous model's prediction.

- **Bagging**: In bagging, multiple models of the same type (i.e., using the same algorithm) are trained. Each model is trained on a different sample of the data, not the entire dataset. The final result is determined by averaging (for regression) or using majority voting (for classification).

- **Boosting**: In boosting, different models are connected in series. The error made by one model is passed on to the next model in the series, which attempts to correct it. This process continues, with each subsequent model focusing on the errors of the previous one.

---
---

## Voting
We are given 3 models, each having an accuracy of 0.7.

                                        1
                                      /   \
                                    /       \
                                  /           \
                                /               \
                              /                   \
                            0.7                   0.3
                            / \                   / \
                          /     \               /     \
                        /        \             /       \
                       0.7       0.3        0.7         0.3
                      / \        / \        / \         / \
                   0.7   0.7   0.7  0.3   0.7  0.3    0.7  0.3
                    ✔    ✔     ✔         ✔            


Final Accuracy:

$0.7 \times 0.7 \times 0.7 + 0.7 \times 0.7 \times 0.3 + 0.7 \times 0.3 \times 0.7 + 0.3 \times 0.7 \times 0.7 = 0.784$

### Types:
- **Hard Voting:** The output with the highest frequency is selected as the final result, i.e., argmax().
- **Soft Voting:** We add probabilities of each class obtained from models and then select the class with the highest value.

---
---

## Stacking
![](https://i.ibb.co/kh92jNk/Untitled.png)

#### 1. Hold Out Method (Blending)
- Split the training data into two parts.
- Train base models on the first part.
- Use the second part to generate predictions using stacked models, which are used as input for the meta-model.
- The meta-model is trained on these predictions.

#### 2. K-Fold Approach (Stacking)
- Split the training data into K folds.
- Train K models of the same type, each leaving out one fold for predictions.
- Predictions from the out-of-fold data are used to train the meta-model.
- The meta-model is trained on the stacked predictions.
- Finally, the base models are retrained on the entire dataset.

#### Multi-Layered Stacking
![](https://i.ibb.co/6rfBBmS/Untitled.png)

---
---

## Bagging Techniques
- **Row Sampling with Replacement:** Standard bagging technique where data is sampled with replacement.
- **Pasting:** Row sampling without replacement.
- **Random Subspaces:** Column sampling, which can be done with or without replacement.
- **Random Patches:** Both row and column sampling are performed, with or without replacement.

  **Out-of-Bag (OOB) Error:** Approximately 37% of samples are not used for model training, so this data can be used for testing the model.
  
---
---
# Ada Boosting

1. **Initial Weights**: For a dataset with $n$ samples, initialize the weight for each row/sample as $\frac{1}{n}$.
   - $w_i = \frac{1}{n}, \quad \text{for all } i \text{ where } i \text{ is the row number}$

   |x|y|$w_i$|
   |-|-|-----|
   | | |$\frac{1}{n}$|

2. **Train a Weak Learner**: Train a decision tree of depth 1 (also known as a decision stump) using the current weights.
3. **Predictions**: Use the trained decision stump to make predictions on the training data.

   |x|y|$w_i$        |$\hat{y}$|
   |-|-|-------------|---------|
   | | |$\frac{1}{n}$|         |

4. **Error Calculation**: Calculate the error $\epsilon$ of the stump, which is the sum of the weights of the misclassified samples:
   - $\epsilon =  w_i \cdot I$. Here, $I : (\hat{y}_i != y_i)$ is the indicator function that returns 1 if the prediction is incorrect and 0 if correct.

   |x|y|$w_i$        |$\hat{y}$|$\epsilon$|
   |-|-|-------------|---------|----------|
   | | |$\frac{1}{n}$|         |          |


5. **Performance of Stump $\alpha$**: Calculate the performance of the stump (also called the weight of the weak learner):
   - $\alpha = \frac{1}{2} \log \left(\frac{1 - \sum \epsilon}{\sum \epsilon}\right)$
  

6. **Update Weights**: Update the weights of the samples based on their prediction outcome:
   - If the prediction is correct: $w_i^{\text{new}} = w_i \cdot e^{-\alpha}$
   - If the prediction is incorrect: $w_i^{\text{new}} = w_i \cdot e^{\alpha}$

   |x|y|$w_i$        |$\hat{y}$|$\epsilon$|$w_i^{\text{new}}$|
   |-|-|-------------|---------|----------|------------------|
   | | |$\frac{1}{n}$|         |          |                  |

7. **Normalize Weights**: Normalize the updated weights so that they sum to 1: $w_i^{\text{new normal}} = \frac{w_i^{\text{new}}}{\sum_{j=1}^n w_j^{\text{new}}}$

   |x|y|$w_i$        |$\hat{y}$|$\epsilon$|$w_i^{\text{new}}$|$w_i^{\text{new normal}}$|
   |-|-|-------------|---------|----------|------------------|-------------------------|
   | | |$\frac{1}{n}$|         |          |                  |                         |

8. **Make Bins**: Create bins corresponding to the normalized weights. The bins are cumulative sums of the weights, which will be used to sample the data points for the next iteration.

$bin_i=(a_i,b_i) = (b_{i-1},b_{i-1}+w_i^{\text{new normal}})$


   |x|y|$w_i$        |$\hat{y}$|$\epsilon$|$w_i^{\text{new}}$|$w_i^{\text{new normal}}$|$bin_i=(a_i,b_i)$|
   |-|-|-------------|---------|----------|------------------|-------------------------|------|
   | | |$\frac{1}{n}$|         |          |                  |                         |      |

9. **Generate Random Numbers**: Generate random numbers between 0 and 1. Each random number corresponds to a bin, and the row whose bin it falls into is selected for training the next weak learner.

10. This process is repeated for a specified number of iterations or until a desired accuracy is achieved and for iteration make sure to use to use $w_i = \frac{1}{n}$.

11. The final model is a weighted sum of all the weak learners. $H(x) = \text{sign} \left( \sum_{n=1}^{N} \alpha_n \cdot h_n(x) \right)$   

---
---

# Gradient Boosting


$Result = Model_0(X) + \eta \cdot Model_1(X) + \eta \cdot Model_2(X) + \dots + \eta \cdot Model_n(X)$

### Gradient Boosting for regression

- ALGORITHM:
   ```python
   prediction = mean(y)  # Initial prediction, starting with the mean of the target values
   models_list.append(Mean_Model)  # Store the initial model
   
   for i in range(1, n_estimators):  # Loop to build subsequent models
       residual = y - prediction  # Compute residuals
       tree = tree.fit(X, residual)  # Train a new model on residuals
       models_list.append(tree)  # Add the trained model to the list
       prediction += η * tree.predict(X)  # Update the prediction with scaled predictions from the new model
   
   result = models_list[0](X) + models_list[1](X) + models_list[2](X)M_0(X) + ...
   ```

---

### Gradient Boosting for Classification

1. **Initialize Log Odds**:
   - Compute the initial log odds:

        **$\text{log odds} = \ln\left(\frac{\text{Count of Ones}}{\text{Count of Zeros}}\right)$**
     
   - Append the initial log odds to the `models_list` as the first model:

     modelsList.append(log_odds)

2. **Loop Over Each Estimator**:
   - For each $i$ from 1 to $n_{\text{estimators}}$:
     1. Calculate the initial probability:

        **$\text{prob} = \frac{1}{1 + e^{-\text{log odds}}}$**
        
     3. Calculate residuals for the current predictions: 

        $\text{residual} = y - \text{prob}$
        
     5. Train a weak learner on residuals: 

        WeakLearner = tree.fit(X, residual)
        
     6. Identify the leaf node number for each data point in the tree.

     7. For each leaf node, compute log loss:

        $\text{log loss} = \frac{\sum \text{Residual}}{\sum[\text{PrevProb} \times (1 - \text{PrevProb})]}$

        
     8. Append the trained model (with calculated log losses in the leaf nodes) to `models_list`: modelsList.append(tree)

     9. For each point, update `log_loss` by adding the weighted log loss from the new tree:
        
        $\text{log loss} += \eta \cdot (\text{log loss from tree})$

4. **Calculate Final Log Loss Prediction**:
   - Aggregate the log losses from each model in `models_list`: $\text{log loss} = modelsList [0] (X) + \eta \cdot modelsList [1] (X) + \eta \cdot modelsList [2] (X) + \dots$

5. **Convert Log Loss to Final Probability**:
   - $\text{prob} = \frac{1}{1 + e^{-\text{log odds}}}$

---

### ALGORITHM(applicale for every Loss Fun)

Given:
- A dataset $\left( (x_i, y_i) \right)_{i=1}^n$.
- A differentiable loss function $L(y, F(x))$ where $L(y, F(x)) = \frac{1}{2} (y - F(x))^2$, which is the squared error loss.
- A maximum number of boosting iterations $M$.

The goal is to build an additive model $f_M(x)$ in a way that minimizes the loss function $L(y, f(x))$ over the training set.

### Steps

1. **Initialize the Base Model**:
   - Start by initializing the model $f_0(x)$ as the constant that minimizes the loss. For squared error loss, this is the mean of $y$:

     $$f_0(x) = \arg \min_{\gamma} \sum_{i=1}^N L(y_i, \gamma) = \text{Mean}(y)$$

2. **Boosting Loop**:
   - For $m = 1$ to $M$:
   
     a. **Compute Residuals**:
        - For each data point $i$, compute the residual $r_{im}$, which represents the negative gradient of the loss function with respect to $f(x_i)$ evaluated at $f_{m-1}(x)$:

          $r_{im} = -\left( \frac{\partial L(y_i, f(x_i))}{\partial f(x_i)} \right)_{f=f(m-1)} $

          $r_{im} = y_i - f_{m-1}(x_i)$


        - This residual measures the error between the actual $y_i$ and the model prediction $f_{m-1}(x_i)$.

     b. **Fit a Regression Tree**:
        - Fit a regression tree to the targets $r_{im}$, producing terminal regions $R_{jm}$ for $j = 1, 2, \ldots, J_m$, where $J_m$ is the number of terminal nodes (leaves) in the tree.

     c. **Compute Terminal Node Predictions**:
        - For each region $R_{jm}$, compute the optimal value $\gamma_{jm}$ that minimizes the loss over the points in $R_{jm}$. Since the loss function is squared error, this $\gamma_{jm}$ is the average residual for points in $R_{jm}$:
          $$\gamma_{jm} = \arg \min_{\gamma} \sum_{x_i \in R_{jm}} L(y_i, f_{m-1}(x_i) + \gamma)$$
          
        - For squared error loss, $\gamma_{jm}$ is the mean of $r_{im}$ for $x_i \in R_{jm}$.

     d. **Update the Model**:
        - Update $f_m(x)$ by adding the scaled contributions of the fitted tree:
          
          $f_m(x) = f_{m-1}(x) + \eta \sum_{j=1}^{J_m} \gamma_{jm} 1(x \in R_{jm})$
        - Here, $\eta$ is a learning rate that controls the contribution of each tree.

3. **Final Output**:
   - After $M$ iterations, output the final model $f_M(x)$, which is the sum of the initial model and the contributions from all $M$ boosting steps.

---
---
# XGBOOST

#### Why XGBoost
- Flexiblity
   - Cross Platform - For windows, Linux
   - Multiple Language Support - Multiple Programing Lang
   - Integration with other libraries and tools
   - Support all kinds of ML problems - Regression, classification, Time Series, Ranking
- Speed - Almost 1/10th time of normal time taken by other ML algos
   - Parallel Processing - Apply parallel processing in creation of decision tree, **n_jobs = -1**
   - Optimized Data Structures - Store data in column instead of rows 
   - Cache Awareness
   - Out of Core computing - **tree_method = hist**
   - Distributed Computing
   - GPU Support - **tree_method = gpu_hist**
- Performance
   - Regularized Learning Objective
   - Handling Missing values
   - sparsity Aware Spit Finding
   - Efficient Spit Finding(Weighted Quantile Sketch + Approximate Tree Learning) - Instead of spliting on each point in a column we can divide the data in bins 
   - Tree Pruning

---

### XGBoost for Regression 

1. **Initialize with the Mean Model**:
   - Set the initial prediction for all data points as the mean of the target variable, $\text{prediction} = \text{mean}(y)$.
   - Store this initial mean model in `models_list`.

2. **Iterative Training with Trees**:
   - For each iteration $i$ from 1 to `n_estimators`:
     - **Calculate Residuals**: $\text{residual} = y - \text{prediction}$
     - **Build a Decision Tree**:
       - Train a decision tree based on a custom "Similarity Score," defined as:
         $\text{Similarity Score} = \frac{\left(\sum \text{ residuals}\right)^2}{\text{Count of residuals} + \lambda}$
       - For each split in the tree:
         - **Calculate Similarity Score** for the tree nodes.
         - Determine splits based on the criterion where $Gani$ is maximized:
           $Gani = SS_{\text{right}} + SS_{\text{left}} - SS_{\text{parent}}$
         - Select the split that maximizes $Gani$.
       - Set the **output at a node**: $\frac{\sum \text{ Residuals}}{\text{Count of residuals} + \lambda}$
     - **Update Prediction**:
       - Add the tree's prediction, scaled by a learning rate $\eta$, to the cumulative prediction:
         $\text{prediction} += \eta \times \text{tree.predict}(X)$

3. **Final Prediction Aggregation**:
   - Combine predictions from all models (starting with the mean model) in `models_list`:
     $\text{result} =$ models_list\[0\]\(X\) + $\eta.$ models_list\[0\]\(X\) + $\eta.$ models_list\[0\]\(X\) + $\dots$

---
### XGBoost for Classification

1. **Initialize Log Odds**:
    - Compute the initial log odds: **$\text{log odds} = \ln\left(\frac{\text{Count of Ones}}{\text{Count of Zeros}}\right)$**
    - Append the initial log odds to the `models_list` as the first model: modelsList.append(log_odds)

2. **Loop Over Each Estimator**:
    - For each $i$ from 1 to $n_{\text{estimators}}$:
        - Calculate the initial probability: **$\text{prob} = \frac{1}{1 + e^{-\text{log odds}}}$**
        - Calculate residuals for the current predictions: $\text{residual} = y - \text{prob}$
        - **Build a Decision Tree**:
            - Train a decision tree based on a custom "Similarity Score," defined as: $\text{Similarity Score} = \frac{\left(\sum  \text{ residuals}\right)^2}{\sum[\text{PrevProb}×(1-\text{PrevProb})] + \lambda}$
           - For each split in the tree:
                - **Calculate Similarity Score** for the tree nodes.
                - Determine splits based on the criterion where $Gani$ is maximized: $Gani = SS_{\text{right}} + SS_{\text{left}} - SS_{\text{parent}}$
                - Select the split that maximizes $Gani$.
            - Set the **output at a node**: $\text{log loss} = \frac{\sum \text{Residual}}{\sum[\text{PrevProb} \times (1 - \text{PrevProb})] + \lambda}$
        - Append the trained model to `models_list`: modelsList.append(tree)
        - For each point, update `log_loss` by adding the weighted log loss from the new tree: $\text{log loss} += \eta \cdot (\text{log loss from tree})$

3. **Calculate Final Log Loss Prediction**: $\text{Total log loss} = modelsList [0] (X) + \eta \cdot modelsList [1] (X) + \eta \cdot modelsList [2] (X) + \dots$

4. **Convert Log Loss to Final Probability**: $\text{prob} = \frac{1}{1 + e^{-\text{log odds}}}$

---

### Mathmatics For XGBoost

$\mathcal{L}^{(t)} = \sum_{i=1}^n L\left(y_i, f_1(x_i) + f_2(x_i) + \dots + f_t(x_i)\right) + \Omega(f_t(x_i))$

$\mathcal{L}^{(t)} = \sum_{i=1}^n L\left(y_i, \hat{y}^{(t-1)} + f_t(x_i)\right) + \Omega(f_t(x_i))$

- The **loss term** measures how well the predictions match the target values.
- The **regularization term** $\Omega$ controls the complexity of the newly added model $f_t(x_i)$, often defined as: $\Omega(f) = \gamma T + \frac{1}{2} \lambda \|w\|^2,$ where $T$ is the number of leaves, $w$ are the leaf weights, and $\gamma, \lambda$ are regularization hyperparameters.

$\mathcal{L}^{(t)} = \sum_{j=1}^T\left[ {\sum_{i \in I_j} g_i w_j} + \frac{1}{2} {\sum_{i \in I_j} h_i w_j^2}  \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$

$\mathcal{L}^{(t)} = \sum_{j=1}^T\left[ {\sum_{i \in I_j} g_i w_j} + \frac{1}{2} {\sum_{i \in I_j} h_i w_j^2} + \frac{1}{2} \lambda w_j^2 \right] + \gamma T$

$\mathcal{L}^{(t)} = \sum_{j=1}^T\left[ {\sum_{i \in I_j} g_i w_j} + \left( \frac{1}{2} {\sum_{i \in I_j} h_i} + \frac{1}{2} \lambda \right) w_j^2 \right] + \gamma T $

$\frac{\partial \mathcal{L}^{(t)}}{\partial w_j} = \sum_{j=1}^T\left[ {\sum_{i \in I_j} g_i} + \left( \frac{1}{2} {\sum_{i \in I_j} h_i} + \frac{1}{2} \lambda \right) 2 w_j \right] = 0$

for a tree Node

$\left[ {\sum_{i \in I_j} g_i} + \left({\sum_{i \in I_j} h_i} + \lambda \right) w_j \right] = 0$

$w_j = \frac{-\sum_{i \in I_j} g_i}{\left(\sum_{i \in I_j} h_i\right) + \lambda}$ , $L^{(t)} = -\frac{1}{2}\sum_{j=1}^T \frac{(\sum_{i\in I_j} g_i)^2}{\sum_{i \in I_j} h_i + \lambda} + γT$
, where $g_i = \frac{\partial L \left(y_i, \hat{y}_i^{(t-1)}\right)}{\partial \hat{y}_i^{(t-1)}}$, $h_i = \frac{\partial^2 L \left(y_i, \hat{y}_i^{(t-1)}\right)}{\partial \hat{y}_i^{(t-1) 2}}$

for regression: $L_i = \frac{1}{2}(y_i - \hat{y}_i)^2$ , $g_i = \frac{\partial L_i}{\partial \hat{y}_i} = (\hat{y}_i - y_i)$ ,  $h_i=\frac{\partial^2 L_i}{\partial \hat{y}_i^2} = 1$

$w_j = \frac{\sum_{i \in I_j} R_i}{N + \lambda}$ , $L_j = \frac{\sum_{i \in I_j} R_i^2}{N + \lambda}$

---



## Bagging Vs Random Forest

| **Bagging** | **Random Forest** |
|-------------|-------------------|
| In Bagging, feature sampling (or selection) is done before training each decision tree. A subset of features is chosen, and the entire tree uses only this subset of features to make splits. | In Random Forest, feature sampling occurs at each split in the tree. A random subset of features is chosen at each node, and the feature with the best Information Gain or Gini Index is used to make the split. |
| This approach introduces less randomness to individual trees, as the same set of features is used throughout each tree. This can lead to lower variance if the features chosen are highly relevant. | By selecting a different subset of features at each split, Random Forest increases the diversity of the trees, helping to reduce overfitting and increasing model robustness by creating a more diverse "forest" of trees. |

---
---
---

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

### Algorithm

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


#### Types of Linkage Criteria (Ways to Measure Distance Between Clusters)

1. **Single Linkage (Minimum Linkage)**
- **Definition**: Distance between two clusters is the **minimum distance** between any point in one cluster and any point in the other.
- **Characteristics**: 
  - Tends to create elongated, chain-like clusters.
  - **Disadvantage**: Highly sensitive to noise or outliers.

2. **Complete Linkage (Maximum Linkage)**
- **Definition**: Distance between two clusters is the **maximum distance** between any point in one cluster and any point in the other.
- **Characteristics**: 
  - Tends to form compact, spherical clusters.
  - **Disadvantage**: May break large clusters into smaller sub-clusters.

3. **Average Linkage**
- **Definition**: Distance between two clusters is the **average distance** between all pairs of points, with one point from each cluster.
- **Characteristics**: 
  - Balances the traits of single and complete linkage.
  - Less sensitive to outliers compared to single linkage.

4. **Ward's Method**
- **Definition**: Measures the increase in variance when clusters are merged by calculating the **difference in squared distance sums** before and after merging clusters.
- **Characteristics**: 
  - Creates compact, spherical clusters.
  - Minimizes within-cluster variance at each step.


### Finding the Ideal Number of Clusters

1. **Plot the Dendrogram**
2. **Cut the Dendrogram Horizontally**
   - Visually inspect the dendrogram and make a horizontal cut at a certain height to define the number of clusters.

3. **Find the Longest Vertical Line**
   - Identify the longest vertical line that does not intersect with any other line, indicating the biggest distance between merged clusters and a natural division.

4. **Determine the Number of Clusters**
   - The ideal number of clusters corresponds to the number of clusters below the horizontal cut through the longest vertical line.
  
---
---


# NLP

## Text Preprocessing Steps

- **Lowercasing**  
   Convert all text to lowercase for uniformity.
- **Remove HTML Tags**  
   Eliminate HTML tags like `<div>` or `<p>` to retain only the plain text.
- **Remove URLs**  
   Strip out any web links from the text.
- **Remove Punctuation**  
   Remove punctuation marks to simplify the text.
- **Chat Word Treatment**  
   Replace common chat abbreviations or slang (e.g., 'u' → 'you', 'r' → 'are').
- **Spelling Correction**  
   Correct misspelled words to their standard forms.
- **Removing Stop Words**  
   Exclude common words like "and", "is", and "the" that do not contribute much meaning.
- **Handling Emojis**  
   Remove or replace emojis with their textual description.
- **Tokenization**  
   - **Word Tokenization**: Break the text into individual words.  
   - **Sentence Tokenization**: Divide the text into sentences.
- **Stemming**  
    Reduce words to their root forms, even if the resulting word lacks meaning (e.g., "running" → "run").
- **Lemmatization**  
    Reduce words to their meaningful base forms (e.g., "better" → "good").

---
---

## Text Representation / Text Embeddings

#### **Common Terms**:

- **Corpus**: A collection of text data used for analysis or training models.
- **Vocabulary**: The unique set of words or tokens in the corpus.
- **Document**: A single piece of text (e.g., a sentence, paragraph, or article) in the corpus.
- **Word**: An individual token from the vocabulary.

---

### One Hot Encoding
- **Steps**:
    - Identify the **Vocabulary** from the corpus.
    - Represent each word using a sparse vector based on the vocabulary, with a single "1" indicating the word's presence.

- **Example**:

    | Document | Content                      |
    |----------|------------------------------|
    | D1       | people watch campusx         |
    | D2       | campusx watch campusx        |
    | D3       | people write comment         |
    | D4       | campusx write comment        |

    Vocabulary = [people, watch, campusx, write, comment]

    | Document | Content                      | Vector                               |
    |----------|------------------------------|--------------------------------------|
    | D1       | people watch campusx         | \[ [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0] \] |
    | D2       | campusx watch campusx        | \[ [0,0,1,0,0], [0,1,0,0,0], [0,0,1,0,0] \] |
    | D3       | people write comment         | \[ [1,0,0,0,0], [0,0,0,1,0], [0,0,0,0,1] \] |
    | D4       | campusx write comment        | \[ [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1] \] |

- Pocs:
    - Intutive
    - Easy Implementation
- Cons:
    - Sparsity
    - No Fixed Size
    - Out Of Vocabulary
    - No campturing of semantic

---

### Bag Of Word(BOW)

- **Steps**:
  - Identify the **Vocabulary**.
  - Represent each document as a fixed-size vector where each unit is the count of a word in the document.
- **Example**:

    | Document | Content                      |
    |----------|------------------------------|
    | D1       | people watch campusx         |
    | D2       | campusx watch campusx        |
    | D3       | people write comment         |
    | D4       | campusx write comment        |

    Vocabulary = [people, watch, campusx, write, comment]

    | Document | Content                |Vector     |Binary Vector|
    |----------|------------------------|-----------|-------------|
    | D1       | people watch campusx   |[1,1,1,0,0]|[1,1,1,0,0]  |
    | D2       | campusx watch campusx  |[0,1,2,0,0]|[0,1,1,0,0]  |
    | D3       | people write comment   |[1,0,0,1,1]|[1,0,0,1,1]  |
    | D4       | campusx write comment  |[0,0,1,1,1]|[0,0,1,1,1]  |

- Pocs:
    - Intutive
    - Easy Implementation
    - Fixed Size
- Cons:
    - Sparsity
    - Out Of Vocabulary
    - Ordering Get Changed
    - No campturing of semantic

---

### N Gram

- **Steps**:
  - Build a vocabulary using **N-word combinations**.
  - Represent each document as a vector where each unit indicates the count of N-grams.

- **Example**:

    | Document | Content                      |
    |----------|------------------------------|
    | D1       | people watch campusx         |
    | D2       | campusx watch campusx        |
    | D3       | people write comment         |
    | D4       | campusx write comment        |

   
    Vocabulary = [people watch, watch campusx, campusx watch, watch campusx, people write, write comment, campusx write]  

    ### Updated Representation:  

    | Document | Content                | Vector         |
    |----------|------------------------|---------------|
    | D1       | people watch campusx   | \[1, 1, 0, 0, 0, 0, 0\] |
    | D2       | campusx watch campusx  | \[0, 0, 1, 1, 0, 0, 0\] |
    | D3       | people write comment   | \[0, 0, 0, 0, 1, 1, 0\] |
    | D4       | campusx write comment  | \[0, 0, 0, 0, 0, 1, 1\] |


- Pocs:
    - Able of campturing of semantic
    - Intutive
    - Easy Implementation
    - Fixed Size
- Cons:
    - Dimension Increses
    - Out Of Vocabulary

---

### TF-IDF (Term Frequency-Inverse Document Frequency)

- **Steps:**
    - Apply Bag of Words:
    - Calculate Term Frequency(Tf): $TF(d, t) = \frac{\text{Number of occurrences of term } t \text{ in document } d}{\text{Total number of terms in document } d}$
    - Calculate Inverse Document Frequency (IDF): $IDF(t) = \ln\left(\frac{\text{Total number of documents in the corpus}}{\text{Number of documents containing term } t}\right)$
    - Compute TF-IDF Weight: $W(d, t) = TF(d, t) \times IDF(t)$
- **Example**:

    | Document | Content                      |
    |----------|------------------------------|
    | D1       | people watch campusx         |
    | D2       | campusx watch campusx        |
    | D3       | people write comment         |
    | D4       | campusx write comment        |

    Vocabulary = [people, watch, campusx, write, comment]

    **BOW**->
    | Document | Content                |Vector     |
    |----------|------------------------|-----------|
    | D1       | people watch campusx   |[1,1,1,0,0]|
    | D2       | campusx watch campusx  |[0,1,2,0,0]|
    | D3       | people write comment   |[1,0,0,1,1]|
    | D4       | campusx write comment  |[0,0,1,1,1]|

    **Tf**->
    | Document | people | watch | campusx | write | comment |
    |----------|--------|-------|---------|-------|---------|
    | D1       | 0.333  | 0.333 | 0.333   | 0.000 | 0.000   |
    | D2       | 0.000  | 0.333 | 0.667   | 0.000 | 0.000   |
    | D3       | 0.333  | 0.000 | 0.000   | 0.333 | 0.333   |
    | D4       | 0.000  | 0.000 | 0.333   | 0.333 | 0.333   |

    **IDF**->
    | Term      | people | watch | campusx | write | comment |
    |-----------|--------|-------|---------|-------|---------|
    | IDF Value | 0.693  | 0.693 | 0.287   | 0.693 | 0.693   |

    **Final TF-IDF (W) Matrix**->
    
    | Document | people | watch | campusx | write | comment |
    |----------|--------|-------|---------|-------|---------|
    | D1       | 0.231  | 0.231 | 0.096   | 0.000 | 0.000   |
    | D2       | 0.000  | 0.231 | 0.191   | 0.000 | 0.000   |
    | D3       | 0.231  | 0.000 | 0.000   | 0.231 | 0.231   |
    | D4       | 0.000  | 0.000 | 0.096   | 0.231 | 0.231   |

    **Final TF-IDF matrix**->
    
    | Document | people | watch | campusx | write | comment |
    |----------|--------|-------|---------|-------|---------|
    | D1       | 0.231  | 0.231 | 0.096   | 0.000 | 0.000   |
    | D2       | 0.000  | 0.231 | 0.382   | 0.000 | 0.000   |
    | D3       | 0.231  | 0.000 | 0.000   | 0.231 | 0.231   |
    | D4       | 0.000  | 0.000 | 0.096   | 0.231 | 0.231   |

- Procs
    - Information Retrival System
- Cons:
    - Sparsity
    - Out Of Vocabulary
    - Ordering Get Changed
    - No campturing of semantic

---

### Word2Vec

#### CBOW

- Make a window of odd size. Let the window size be 3.
- The context words (word1, word3) are used to predict the target word (word2) ->  word1 ....?....  word3
- Convert the words to one-hot encoding vectors.
- feed it to neuron network given below 

![](https://github.com/ParitKansal/photos/blob/main/CBOW.png)


#### Skip Gram
- Make a window of odd size. Let the window size be 3.
- The target word (word2) is used to predict the context words (word1, word3). -> ....?.... word2 ....?....
- Convert the words to one-hot encoding vectors.
- Feed it to neuron network given below

![](https://github.com/ParitKansal/photos/blob/main/SkipGram.png)

---
