
| **Metric**      | **Formula**                                                                             | **Description**                             |
|-----------------|-----------------------------------------------------------------------------------------|---------------------------------------------|
| **Accuracy**    | $\frac{TP + TN}{TP + TN + FP + FN}$                                                     | Greater values indicate better performance. |
| **Precision**   | $\frac{TP}{TP + FP} = \frac{\text{True Positive}}{\text{Predicted Positive}}$           | Greater values indicate better performance. |
| **Recall**      | $\frac{TP}{TP + FN} = \frac{\text{True Positive}}{\text{Real Positive}}$                | Greater values indicate better performance. |
| **F1-score**    | $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$                             | Greater values indicate better performance. |
| **Log Loss**    | $- \frac{1}{n} \sum \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$ | Lower values indicate better performance.   |






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
