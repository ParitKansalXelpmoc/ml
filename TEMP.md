
## Topics
- [Linear Regression](https://github.com/ParitKansal/ml/edit/main/TEMP.md#linear-regression)
- 

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

## Polynomial Regression

Suppose we have three features and we want apply degree 2 polynonial features then calculate or make ney features -> $x^2 , y^2, z^2, xy, xz, yz$. Now apply normal linear regression.

---








---
