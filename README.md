# XGBOOST

#### Why Gradient Boosting
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
---

# Gradient Boosting
$Result = Model_0(X) + \eta \cdot Model_1(X) + \eta \cdot Model_2(X) + ... + \eta \cdot Model_n(X)$

```python
prediction = mean(y)
models_list.append(Mean_Model)

for i in rangle(n_estimators):
   residual = y - prediction
   tree = tree.fit(X, residual)
   tree_list.append(tree)
   prediction += η*tree.predict(X)
```



$Prediction = mean(y)$
$ residuals = y - predictions$
$tree = self._fit_tree(X, residuals)$
$model_list.append(tree)$




|$x$|$y$|$\hat{y} = f(x, \bar{y})$|$R_1 = y - \hat{y}$|
|---|---|-------------------|-------------|
















$\sum y_i -f_0(x)-y = 0$

$γ_{jm} = argmin_γ \sum_{x_i ∈ R_{jm}} L(y_i, f_{m-1}(x_i) + γ)$



Input: training data set $\left(( x_i, y_i )\right)_{i=1}^n$, a differentiable loss function $L\left( y, F(x)\right)$, number of iterations M
   - Suppose Loss function, $L\left( y, F(x)\right) = \frac{1}{2} \left(y-γ\right)^2$
1. Initilize $f_0(x) = arg min_{γ} \sum^{N}_{i=1} L\left( y_i  , γ \right)$
   - $f_0(x) = Mean(y)$
2. For m = 1 to M :

   (a). For $i = 1, 2, ... N$ compute $r_{im} = - \left(\frac{\partial L(y_i, f(x_i))}{\partial f(x_i)}\right)_{f=f_m-1}$ 

      - $r_{im} = \left(y_i - f(x_i)\right)_{f=f_m-1}$

   (b). Fit a regression tree to targets $r_{im}$ giving terminal regions $R_{jm}$, $j = 1, 2, 3 ... J_m$ on $\left(( x_i, r_{im} )\right)_{i=1}^n$
   
   (c). For $j = 1, 2, 3 ... J_m$ compute
   

























$Prediction = M_1 + \eta_{1}M_2 + \eta_{2}M_3 + ... $

$Pred_1 = Mean(Y)$
$Res_1 = Y - Pred_1$

$M2 = Model(X, Res_1)$
$ Pred_2 = Pred_1 + M2(X)$
$Res_2 = Y - Pred2$

$M3 = Model(X, Res_2)$
$ Pred_3 = Pred_2 + M3(X)$
$Res_3 = Y - Pred3$















$residual_n = actual - ({pred_1} + \eta_{2}{pred_2} + \eta_{3}{pred_3} + ... + \eta_{n-1}{pred_{n-1}})$, where $n > 0$

$pred_n = Model(X, residual_{n-1})$

$pred_n = Avg(Y)$


residual_n = actual - prediction_n
residual_n = actual - (pred_0 + eta

- Train model (input, y) which gives average of y
- pred1
- residual1 = y - pred1
- Train model (input, residual1)
- pred2
- residual2 = residual1 - pred2















---
---





# SVM (Support Vector Machine)
![](https://i.postimg.cc/N0YydtDp/Untitled.png)

- The aim of this is to maximise the distance between $π_+$ and $π_-$ and minimise the distance distance of wrong output pints to there repective correct margin plane.

$\arg\min_{w*, b*} \left( Margin Error + Classification Error\right)$

$\arg\min_{w*, b*} \left( \frac{1}{2} ||w||^2 + C \sum_{i=1}^n \zeta_i \right)$

### where:
- **$w$** : Weight vector.
- **$b$** : Bias term.
- **$C $** : Regularization parameter.
- **$\frac{1}{2} ||w||^2 $** : Regularization term that penalizes large weights.
- **$\sum_{i=1}^n \zeta_i $** : Sum of the slack variables $𝜁_𝑖$​, which represent the amount by which each data point deviates from the correct classification margin.

This objective function seeks to minimize a combination of the margin (through the regularization term) and the misclassification error (through the slack variables). The goal is to find the optimal $w*$ and $b*$ that achieve this balance.

**We use Kernels for non linear seperable data**






---
---






# Ensemble Learning
Based on the concept of the wisdom of the crowd, decisions made by multiple models have a higher chance of being correct. Ensemble learning helps convert a low-bias, high-variance model into a low-bias, low-variance model because the random/outliers are distributed among various models instead of going to a single model.

#### Voting Ensemble
In this approach, different models of various types, such as SVM, Decision Tree, and Logistic Regression, calculate results. For classification tasks, the output with the highest frequency is selected as the final result, or we add probabilities of each class obtained from models. For regression tasks, the mean of the outputs is calculated.

#### Stacking
In stacking, different models of various types, such as SVM, Decision Tree, and Logistic Regression, are used to calculate results. The outputs of these models are then used to train a final model, such as KNN, to obtain the final output. This final model effectively assigns a weight to each previous model's prediction.

#### Bagging
In bagging, multiple models of the same type (i.e., using the same algorithm) are trained. Each model is trained on a different sample of the data, not the entire dataset. The final result is determined by averaging (for regression) or using majority voting (for classification).

#### Boosting
In boosting, different models are connected in series. The error made by one model is passed on to the next model in the series, which attempts to correct it. This process continues, with each subsequent model focusing on the errors of the previous one.

## Bagging Techniques
- **Row Sampling with Replacement:** Standard bagging technique where data is sampled with replacement.
- **Pasting:** Row sampling without replacement.
- **Random Subspaces:** Column sampling, which can be done with or without replacement.
- **Random Patches:** Both row and column sampling are performed, with or without replacement.

  **Out-of-Bag (OOB) Error:** Approximately 37% of samples are not used for model training, so this data can be used for testing the model.

## Decision tree vs. Bagging
| **Bagging** | **Decision Tree** |
|-------------|-------------------|
| Features are selected before training the decision tree, i.e., feature sampling is done at the tree level. | Some features are selected randomly at each node, and Information Gain (or Gini Index) is calculated for these feature to decide the best split. |
| It introduces less randomness into the model as the same set of features is used across the entire tree. | It introduces more randomness as the features are selected at each node, which can lead to different splits and trees. |

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
   - $\epsilon = \sum_{i=1}^n w_i \cdot I(\text{pred}_i \neq \text{actual}_i)$. Here, $I$ is the indicator function that returns 1 if the prediction is incorrect and 0 if correct.

   |x|y|$w_i$        |$\hat{y}$|$\epsilon$|
   |-|-|-------------|---------|----------|
   | | |$\frac{1}{n}$|         |          |


5. **Performance of Stump $\alpha$**: Calculate the performance of the stump (also called the weight of the weak learner):
   - $\alpha = \frac{1}{2} \log \left(\frac{1 - \epsilon}{\epsilon}\right)$
  

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

   |x|y|$w_i$        |$\hat{y}$|$\epsilon$|$w_i^{\text{new}}$|$w_i^{\text{new normal}}$|$bins$|
   |-|-|-------------|---------|----------|------------------|-------------------------|------|
   | | |$\frac{1}{n}$|         |          |                  |                         |      |

9. **Generate Random Numbers**: Generate random numbers between 0 and 1. Each random number corresponds to a bin, and the row whose bin it falls into is selected for training the next weak learner.
10. This process is repeated for a specified number of iterations or until a desired accuracy is achieved and for iteration 1 to set 9 are folloewd and make sure to use to use $w_i$ not $w_i^{\text{new}}$.
11. The final model is a weighted sum of all the weak learners. $H(x) = \text{sign} \left( \sum_{t=1}^{T} \alpha_t \cdot h_t(x) \right)$   







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


# Cure Algorithm

### **Pass 1: Initial Clustering, Outlier Removal, and Selection of Representative Points**

1. **Random Sampling and Initial Clustering**
   - Select a random sample of data points that fit in main memory.
   - Cluster these sampled points to form initial clusters (e.g., using k-means or a similar clustering algorithm).

2. **Outlier Detection and Removal**
    - For each initial cluster, calculate the distance of each point from the cluster centroid.
    - Identify outliers as points that are significantly farther from the centroid than the average distance.
    - Remove these outliers from the dataset. Outliers can either be stored separately for analysis or discarded based on requirements.

3. **Selection of Scattered Representative Points**
    - For each cluster (now without outliers), select `c` scattered points within the cluster (e.g., `c = 4`).
    - These points act as representatives of each cluster.

4. **Shrinking Representative Points Towards the Centroid**
    - Move each representative point closer to the cluster’s centroid by a factor of `α`, where `0 < α < 1`.
    - This "shrinking" adjusts the representative points, enhancing their alignment with the cluster's core.

5. **Cluster Merging Using the Minimum-Distance (dmin) Approach**
    - Apply a minimum-distance (`dmin`) criterion to merge clusters that have representative points within a close distance.
    - This step decreases the number of clusters, making the dataset more manageable for Pass 2.


### **Pass 2: Refining Clusters with Iterative Merging**

1. **Updating Representatives After Each Merge**
    - After each cluster merge, designate the newly combined points as representatives of the new cluster.
    - Continue merging clusters based on the updated representative points.

2. **Stopping Condition**
    - The merging process continues until the specified number of clusters, `k`, is achieved.
-------
--------

|||
|-|-|
|Select python env in anaconda
| Launch VSCode
| Select folder
| Select -> ‘Shift’ + ‘Ctrl’ + ‘p’
| Select the python env
| In cmd terminal type -> py -3 -m venv venv
| Select->’Shift’+’Ctrl’+’p’
| Type path of virtual env -> C:\Users\shrut\OneDrive\Desktop\fast - api\venv\Scripts\python.exe
| In cmd type-> venv\Scripts\activate.bat
| In cmd type-> pip install fastapi[all]
| In cmd type -> pip freeze
| In main.py type code
|In cmd type -> uvicorn main:app |In cmd type -> uvicorn main:app – – reload|
|Copy url generated and paste it in browser |
|To stop the server press Ctrl + ‘C’ |
