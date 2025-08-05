

# Breast Cancer Prediction using Machine Learning

## Objective
To develop a machine learning model that predicts whether a breast tumor is malignant or benign based on diagnostic features.

## Project Description

# **# Project Title:** - **Breast Cancer Prediction Using Logistic Regression**

#**PROBLEM STATEMENT AND AIM OF THE PROJECT:**

Breast cancer is one of the most common cancers among women worldwide. Early diagnosis and accurate classification of breast cancer tumors as either malignant (cancerous) or benign (non-cancerous) are critical for effective treatment and improved survival rates.





**This project aims to develop a machine learning-based classification system using Logistic Regression to predict the nature of breast tumors based on input features derived from digitized images of fine needle aspirates (FNA) of breast masses.**





By leveraging the Breast Cancer Wisconsin Diagnostic Dataset from sklearn.datasets, the objective is to:





--> Train a logistic regression model on labeled data,



--> Evaluate its accuracy,



--> Build a predictive system that can classify new tumor samples,



--> Aid medical professionals in early detection and diagnosis.





**This solution provides a lightweight, interpretable, and efficient tool suitable for medical screening applications.**

**# Technologies:**

Python, NumPy, Pandas, Scikit-learn

# **Project Description:**

This project involves predicting whether breast cancer is malignant or benign using logistic regression.

# **Data Collection & Loading:**

- Import the Breast Cancer dataset from sklearn.

- Load the dataset into a pandas DataFrame with feature_names as columns.

# **Exploratory Data Analysis (EDA):**

- Display the first five rows of the dataset using .head().

- Add the target column to the DataFrame and display the last five rows using .tail().

- Analyze the dataset:

- Use .shape to check the number of rows and columns.

- Use .info() for an overview of column types and non-null values.

- Check for missing values using .isnull().sum().

- Display summary statistics with .describe().

- Analyze the target variable distribution using .value_counts().

# **Summary of EDA Results**

The dataset contains 569 samples and 31 features (30 features and 1 target variable).



There are no missing values in the dataset.



The features are numerical and represent characteristics of the breast cell nuclei.



The target variable 'label' is binary, with 357 samples labeled as benign (1) and 212 samples labeled as malignant (0).



The summary statistics provide insights into the distribution of each feature, including mean, standard deviation, min, max, and quartiles.



# **Data Preprocessing**

# ***Train-Test Split***

# **Model Training**

# **Model Evaluation**

# **Model Evaluation Results Explained:**

Training Accuracy: 0.9582417582417583 Test Accuracy: 0.9649122807017544



**Accuracy:** Both training and test accuracies are high, indicating that the model performs well on both seen and unseen data. This suggests that the model is n**ot overfitting.**



Precision: 0.9722222222222222 Recall: 0.9722222222222222 F1 Score: 0.9722222222222222



**Precision:**

Of all the instances predicted as positive (benign), 97.22% were actually positive.



**Recall:** Of all the actual positive instances (benign), 97.22% were correctly identified by the model.

F1 Score: The F1 score is a harmonic mean of precision and recall, providing a balanced measure of the model's performance. A high F1 score indicates a good balance between precision and recall.



**Confusion Matrix:**



The confusion matrix visualizes the performance of a classification model.



True Positives (TP): The number of instances correctly predicted as positive (bottom right: 70).

True Negatives (TN): The number of instances correctly predicted as negative (top left: 40).

False Positives (FP): The number of instances incorrectly predicted as positive (top right: 2). These are Type I errors.

False Negatives (FN): The number of instances incorrectly predicted as negative (bottom left: 2). These are Type II errors.

In this confusion matrix:



The model correctly identified 40 malignant cases (TN).

The model correctly identified 70 benign cases (TP).

The model incorrectly predicted 2 malignant cases as benign (FN).

The model incorrectly predicted 2 benign cases as malignant (FP).

ROC Curve and AUC:



**ROC Curve (Receiver Operating Characteristic Curve):** This plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings. It shows the trade-off between sensitivity (TPR) and specificity (1-FPR).



**AUC (Area Under the Curve):** The AUC represents the degree or measure of separability. It tells how much the model is capable of distinguishing between classes. A higher AUC means the model is better at predicting 0s as 0s and 1s as 1s.



**AUC Value:** The AUC value of 1.00 indicates a perfect model that can perfectly distinguish between malignant and benign cases based on the current data and threshold.



**In summary:** The logistic regression model performed exceptionally well on this dataset, achieving high accuracy, precision, recall, and F1 scores on both the training and test sets. The confusion matrix shows a low number of misclassifications, and the ROC curve with an AUC of 1.00 indicates excellent discriminatory power.

# **#7. Building a Predictive System:**

## Key Steps
- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Model training using various algorithms
- Model evaluation and performance comparison

## Tools & Libraries Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

## File
- `ML_project2_breast_cancer_prediction_shati_biswas.ipynb`: The Jupyter notebook containing all code and analysis.

## How to Run
1. Clone this repository or download the project files.
2. Install the required Python libraries.
3. Open the notebook and run the cells in order.
