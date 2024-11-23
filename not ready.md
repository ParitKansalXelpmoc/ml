# **Handling Imbalanced Data**

### 1. **Undersampling and Oversampling**  
![](https://github.com/ParitKansal/ml/blob/main/photos/1_7xf9e1EaoK5n05izIFBouA.webp)

---

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
