
$\mathcal{L}^{(t)} = \sum_{j=1}^T\left[ {\sum_{i \in I_j} g_i w_j} + \frac{1}{2} {\sum_{i \in I_j} h_i w_j^2}  \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$

$\mathcal{L}^{(t)} = \sum_{j=1}^T\left[ {\sum_{i \in I_j} g_i w_j} + \frac{1}{2} {\sum_{i \in I_j} h_i w_j^2} + \frac{1}{2} \lambda w_j^2 \right] + \gamma T$

$\mathcal{L}^{(t)} = \sum_{j=1}^T\left[ {\sum_{i \in I_j} g_i w_j} + \left( \frac{1}{2} {\sum_{i \in I_j} h_i} + \frac{1}{2} \lambda \right) w_j^2 \right] + \gamma T $

$\frac{\partial \mathcal{L}^{(t)}}{\partial w_j} = \sum_{j=1}^T\left[ {\sum_{i \in I_j} g_i} + \left( \frac{1}{2} {\sum_{i \in I_j} h_i} + \frac{1}{2} \lambda \right) 2 w_j \right] = 0$

for a tree Node

$\left[ {\sum_{i \in I_j} g_i} + \left({\sum_{i \in I_j} h_i} + \lambda \right) w_j \right] = 0$

$w_j = \frac{-{\sum_{i \in I_j} g_i}} {{\sum_{i \in I_j} h_i} + \lambda} $

1. **First Equation:**

    $\mathcal{L}^{(t)} = \sum_{i=1}^n L\left(y_i, f_1(x_i) + f_2(x_i) + \dots + f_t(x_i)\right) + \Omega(f_t(x_i))$

    $\mathcal{L}^{(t)} = \sum_{i=1}^n L\left(y_i, \hat{y}^{(t-1)} + f_t(x_i)\right) + \Omega(f_t(x_i))$

- The **loss term** measures how well the predictions match the target values.
- The **regularization term** $\Omega$ controls the complexity of the newly added model $f_t(x_i)$, often defined as:
  $\Omega(f) = \gamma T + \frac{1}{2} \lambda \|w\|^2,$
  where $T$ is the number of leaves, $w$ are the leaf weights, and $\gamma, \lambda$ are regularization hyperparameters.



---

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
# Gradient Boosting for Classification

- **Initialize Log Odds**:
    - Compute the initial log odds: **$\text{log odds} = \ln\left(\frac{\text{Count of Ones}}{\text{Count of Zeros}}\right)$**
    - Append the initial log odds to the `models_list` as the first model: modelsList.append(log_odds)

- **Loop Over Each Estimator**:
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

- **Calculate Final Log Loss Prediction**: $\text{Total log loss} = modelsList [0] (X) + \eta \cdot modelsList [1] (X) + \eta \cdot modelsList [2] (X) + \dots$

- **Convert Log Loss to Final Probability**: $\text{prob} = \frac{1}{1 + e^{-\text{log odds}}}$
---

## Bagging Vs Random Forest


| **Bagging** | **Random Forest** |
|-------------|-------------------|
| In Bagging, feature sampling (or selection) is done before training each decision tree. A subset of features is chosen, and the entire tree uses only this subset of features to make splits. | In Random Forest, feature sampling occurs at each split in the tree. A random subset of features is chosen at each node, and the feature with the best Information Gain or Gini Index is used to make the split. |
| This approach introduces less randomness to individual trees, as the same set of features is used throughout each tree. This can lead to lower variance if the features chosen are highly relevant. | By selecting a different subset of features at each split, Random Forest increases the diversity of the trees, helping to reduce overfitting and increasing model robustness by creating a more diverse "forest" of trees. |












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

---
---
---

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
