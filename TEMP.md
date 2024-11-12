
---
---
### Feature Scaling
1. **Standardization**
   - Formula: $x' = \frac{x - \text{mean}(x)}{\sigma}$
   - Standardized data has a mean of 0 and a standard deviation of 1.
   - Useful when features are on different scales; works well with algorithms that assume normal distribution.

2. **Normalization**
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

### Encoding Categorical Data

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
## Encoding numerical features
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

### Outlier Handling Techniques

1. **Trimming (Removing Outliers)**
   - Remove the outliers

2. **Capping**
   - Replace the outliers with a set boundary
---
---
