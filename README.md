# ML
## Ensemble Learning
Based on the concept of the wisdom of the crowd, decisions made by multiple models have a higher chance of being correct. Ensemble learning helps convert a low-bias, high-variance model into a low-bias, low-variance model bcoz the random/outliers are distributed among various models instead of going to a single model.

#### Voting Ensemble
In this approach, different models of various types, such as SVM, Decision Tree, and Logistic Regression, calculate results. For classification tasks, the output with the highest frequency is selected as the final result or we add probablities of each class obtained from models. For regression tasks, the mean of the outputs is calculated.

#### Stacking
In stacking, different models of various types, such as SVM, Decision Tree, and Logistic Regression, are used to calculate results. The outputs of these models are then used to train a final model, such as KNN, to obtain the final output. This final model effectively assigns a weight to each previous model's prediction.

#### Bagging
In bagging, multiple models of the same type (i.e., using the same algorithm) are trained. Each model is trained on a different sample of the data, not the entire dataset. The final result is determined by averaging (for regression) or using majority voting (for classification).

#### Boosting
In boosting, different models are connected in series. The error made by one model is passed on to the next model in the series, which attempts to correct it. This process continues, with each subsequent model focusing on the errors of the previous one.

## Bagging Techniques
- **Row Sampling with Replacement:** Standard bagging technique where data is sampled with replacement.
- **Pasting:** Row sampling without replacement.
- **Random Subspaces:** Column sampling, which can be done with or without replacement.
- **Random Patches:** Both row and column sampling are performed, with or without replacement.

  **Out-of-Bag (OOB) Error:** Approximately 37% of samples are not used for model training, so this data can be used for testing the model.


## Decision tree Vs Bagging

## Voting
We are given 3 models each having accuracy of 0.7 . 

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

Final Accuracy = 0.7×0.7×0.7 + 0.7×0.7×0.3 + 0.7×0.3×0.7 + 0.3×0.7×0.7 = 0.784

### Types-
- **Hard Voting** - the output with the highest frequency is selected as the final result i.e. argmax()
- **Soft Voting** - we add probablities of each class obtained from models and then select the class having gighest value

## Stacking
![](https://i.ibb.co/kh92jNk/Untitled.png)

#### 1. Hold Out Method (Blending)
- Split the training data into two parts.
- Train base models on the first part.
- Use the second part to generate predictions using stacked models, which are used as input for the meta-model.
- The meta-model is trained on these predictions.

#### 2. K-Fold Approach (Stacking)
- Split the training data into K folds.
- Train K models of the same type, each leaving out one fold for Predictions.
- Predictions from the out-of-fold data are used to train the meta-model.
- the meta-model is trained on the stacked predictions.
- Finally, the base models are retrained on the entire dataset.

#### Multi Layered Stacking
![](https://i.ibb.co/6rfBBmS/Untitled.png)

















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




