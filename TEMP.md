
# Topics
- [Linear Regression](#linear-regression)
- [Polynomial Regression](#polynomial-regression)
- [Ridge, Lasso, Elastic Regression](#ridge-lasso-elastic-regression)
- [Logistic regression](#logistic-regression)
   - [Percepton Trick](#1-percepton-trick)
   - [Sigmoid function](#2-sigmoid-function)
   - [Maximum Likelihood](#3-maximum-likelihood)
- [Softmax Regression / Multinomial Logistic Regression](#softmax-regression--multinomial-logistic-regression)
- [KNN](#knn)
- [Naive Bayes Classifier](#naive-bayes-classifier)
   - [Handling Numerical Values](#handling-numerical-values)
- [CART](#cart---classification-and-regression-trees)
   - [Decision Tree](#decision-tree)
   - [Regression Trees](#regression-trees)
- [Feature Importance](#feature-importance-for-decision-tree-like-algos)
- [SVM](#svm-support-vector-machine)
- [Ensemble Learning](#ensemble-learning)
- [Voting Ensemble](#voting)
- [Stacking](#stacking)
- [Bagging](#bagging-techniques)
- [Boosting]
- [Ada Boosting](#ada-boosting)
---
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

$\text{Loss, } L = \sum (y - \hat{y})^2 + \lambda ||w|| = \sum (y - \hat{y})^2 + \lambda (|w_1| + |w_2| + |w_3| + \dots + |w_n|)$

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



