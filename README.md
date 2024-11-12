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
