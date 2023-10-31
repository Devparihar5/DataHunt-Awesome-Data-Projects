<h1 align="center"> <strong> Spam Ham classification </strong> </h1>

In this project, the performance of various **Machine Learning Classifiers** is investigated under a **slight class imbalance** and compared to the performance after oversampling (both resampling the minority and undersampling the majority class samples) in terms of test accuracy and training time.

The models used for the comparative study are:
1. **Multinomial Naive Bayes model** 
2. **Logistic Regression** 
3. **KNN** 
4. **SVM (Linear)**
5. **SVM (RBF)**
6. **Random Forest**

The original and synthetically generated samples are visualised using **PCA** (2D Visualization)

The idea is to perform a comparative analysis for these models and evaluate each of these on the following metrics
1. F1 score
2. F2 score
3. Precision
4. Recall
5. ROCAUC score

The effect of introducing synthetic samples in the training data is then analysed for each classifier.
