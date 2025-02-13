
---

### **Types of Data**  
- **Categorical Data**  
  - **Nominal Data**: Categories without a specific order (e.g., colors, gender, city names).  
  - **Ordinal Data**: Categories with a meaningful order (e.g., rankings, satisfaction levels).  

- **Numerical Data**  
  - **Continuous Data**: Can take any value within a range (e.g., height, temperature).  
  - **Discrete Data**: Can only take specific values (e.g., number of students in a class).  

---

### **Measure of Central Tendency**  
These measures summarize a dataset by identifying the center.  

1. **Mean (Average)**  
   - **Sample Mean (x̄)**: $x̄ = \frac{\sum x_i}{n}$  
   - **Population Mean (μ)**: $μ = \frac{\sum X_i}{N}$  

2. **Median**: The middle value when data is sorted.  

3. **Mode**: The most frequently occurring value in a dataset.  

4. **Weighted Mean**: Mean with different weights assigned to data points.  

5. **Trimmed Mean**: Mean calculated after removing extreme values (outliers).  

---

### **Measure of Dispersion**  
These measures describe the spread of data.  

1. **Range**:  
   - $\text{Range} = \max(X) - \min(X)$  

2. **Variance**: Measures how much the values deviate from the mean.  
   - **Sample Variance (s²)**: $s^2 = \frac{\sum (x_i - x̄)^2}{n-1}$  
   - **Population Variance (σ²)**: $σ^2 = \frac{\sum (X_i - μ)^2}{N}$  

3. **Standard Deviation (Std)**: The square root of variance.  
   - **Sample Std (s)**: $s = \sqrt{s^2}$  
   - **Population Std (σ)**: $σ = \sqrt{σ^2}$  

4. **Coefficient of Variation (CV)**: A relative measure of dispersion.  
   - **Sample CV**: $\text{CV} = \frac{s}{x̄} \times 100\%$  
   - **Population CV**: $\text{CV} = \frac{σ}{μ} \times 100\%$

---

### Quantiles
1. **Quartiles**: Divide the data into four equal parts; Q1 (25th percentile), Q2 (50th percentile or median), and Q3 (75th percentile).
2. **Deciles**: Divide the data into ten equal parts; D1 (10th percentile), D2 (20th percentile), ... D9 (90th percentile).
3. **Percentiles**: Divide the data into 100 equal parts; P1 (1st percentile), P2 (2nd percentile), ... P99 (99th percentile).
4. **Quintiles**: Divide the data into 5 equal parts.

#### **Percentile Calculation**

A percentile is a statistical measure that represents the percentage of observations in a dataset that fall below a particular value. For example, the 75th percentile is the value below which 75% of the observations in the dataset fall.

#### **Formula to calculate the percentile value:**  
$P_L = \frac{P}{100} \times (N + 1)$

**Where:**  
- $P_L$ = the desired percentile value location  
- $N$ = the total number of observations in the dataset  
- $P$ = percentile rank (expressed as a percentage)  

#### **Calculating the 72nd Percentile**  

We will follow the following steps as before to find the **72nd percentile**.

#### **Step 1:** Given Data  
Original dataset:  
**78, 82, 84, 88, 91, 94, 96, 98, 94**

**Sorted Data:**  
**78, 82, 84, 88, 91, 94, 94, 96, 98**  

#### **Step 2:** Determine the Position  
Using the percentile formula:  
$P_L = \frac{72}{100} \times (9 + 1) = 0.72 \times 10 = 7.2$  

This means the 72nd percentile is at position **7.2**, which falls between the **7th and 8th** values.

#### **Step 3:** Identify the Values  
From the sorted data:  
- 7th value = **94**  
- 8th value = **96**  

#### **Step 4:** Compute the 72nd Percentile  
We interpolate between the **7th and 8th** values:  
$\text{Percentile Value} = 94 + 0.2 \times (96 - 94) = 94 + 0.4 = 94.4$  

---

### **Covariance and Correlation**  

#### **1. Covariance (Measures the Direction of Relationship)**  
Covariance measures how two variables change together. It indicates whether an increase in one variable corresponds to an increase or decrease in another.  

- **Formula:**  
  - **Sample Covariance:**  
    $\text{Cov}(X, Y) = \frac{\sum (x_i - x̄)(y_i - ȳ)}{n-1}$
  - **Population Covariance:**  
    $\text{Cov}(X, Y) = \frac{\sum (X_i - μ_X)(Y_i - μ_Y)}{N}$

- **Interpretation:**  
  - **Positive Covariance** → Variables move in the same direction.  
  - **Negative Covariance** → Variables move in opposite directions.  
  - **Zero Covariance** → No relationship between variables.  

However, covariance does not provide a standardized measure, making it difficult to interpret across datasets.

---

#### **2. Correlation (Measures Strength & Direction of Relationship)**  
Correlation standardizes covariance by dividing it by the product of standard deviations.  

- **Formula (Pearson Correlation Coefficient, $r$):**  
  $r = \frac{\text{Cov}(X, Y)}{s_X s_Y}$
  where $s_X$ and $s_Y$ are the standard deviations of $X$ and $Y$.

- **Interpretation (Range: -1 to 1):**  
  - $r = +1$ → Perfect positive correlation  
  - $r = -1$ → Perfect negative correlation  
  - $r = 0$ → No correlation

---

## **Correlation vs. Causation**

The phrase **"correlation does not imply causation"** means that just because two variables are related does not necessarily mean that one causes the other. In other words, a correlation between two variables does not imply that one variable is responsible for the behavior of the other.

### **Example: Firefighters and Fire Damage**  
Suppose there is a **positive correlation** between the number of firefighters present at a fire and the amount of damage caused by the fire. One might be tempted to conclude that the presence of more firefighters causes greater damage. However, this conclusion is flawed.

A third factor, **the severity of the fire**, explains this correlation:  
- **Larger fires** require more firefighters.  
- **Larger fires** also cause more damage.  

Thus, while the number of firefighters and fire damage are correlated, one does not cause the other.

### **Establishing Causation**  
Correlations can provide valuable insights into relationships between variables, but **they cannot establish causality**. To determine causation, additional evidence is needed, such as:  
1. **Controlled Experiments** – Manipulating one variable while keeping others constant.  
2. **Randomized Controlled Trials (RCTs)** – Randomly assigning subjects to control and experimental groups.  
3. **Well-Designed Observational Studies** – Using statistical methods to control for confounding variables.  

---


