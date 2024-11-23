
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

