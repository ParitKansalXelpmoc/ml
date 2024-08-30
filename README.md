Here is the corrected version of your notes on Ensemble Learning:

# ML
## Ensemble Learning
Based on the concept of the wisdom of the crowd, decisions made by multiple models have a higher chance of being correct. Ensemble learning helps convert a low-bias, high-variance model into a low-bias, low-variance model.

### Voting Ensemble
In this approach, different models of various types, such as SVM, Decision Tree, and Logistic Regression, calculate results. For classification tasks, the output with the highest frequency is selected as the final result. For regression tasks, the mean of the outputs is calculated.

### Stacking
In stacking, different models of various types, such as SVM, Decision Tree, and Logistic Regression, are used to calculate results. The outputs of these models are then used to train a final model, such as KNN, to obtain the final output. This final model effectively assigns a weight to each previous model's prediction.

### Bagging
In bagging, multiple models of the same type (i.e., using the same algorithm) are trained. Each model is trained on a different sample of the data, not the entire dataset. The final result is determined by averaging (for regression) or using majority voting (for classification).

### Boosting
In boosting, different models are connected in series. The error made by one model is passed on to the next model in the series, which attempts to correct it. This process continues, with each subsequent model focusing on the errors of the previous one.

## Bagging Techniques
- **Row Sampling with Replacement:** Standard bagging technique where data is sampled with replacement.
- **Pasting:** Row sampling without replacement.
- **Random Subspaces:** Column sampling, which can be done with or without replacement.
- **Random Patches:** Both row and column sampling are performed, with or without replacement.

  **Out-of-Bag (OOB) Error:** Approximately 37% of samples are not used for model training, so this data can be used for testing the model.
