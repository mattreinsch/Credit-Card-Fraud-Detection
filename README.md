# Credit Card Fraud Detection Using Python

![Credit Card Fraud Detection](CreditCard.jpg)

## Introduction
In today’s digital age, credit card fraud poses a significant threat to financial security. To combat this, data science and machine learning offer powerful tools for detecting fraudulent transactions. This blog will walk you through a Python-based approach to credit card fraud detection, leveraging a real-world dataset and various data analysis techniques.

## Step 1: Importing Libraries
First, we need to import the necessary libraries:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
%matplotlib inline
```
These libraries will help us with data manipulation, visualization, and machine learning.

## Step 2: Loading the Dataset
Next, we load the dataset containing credit card transactions:

```python
df = pd.read_csv('creditcard.csv')
print(df.shape)
df.head()
```
The dataset has 31 columns: the transaction features (`V1` to `V28`), `Time`, `Amount`, and the `Class` label indicating whether a transaction is fraudulent (`1`) or not (`0`).

## Step 3: Data Exploration
We explore the dataset to understand its structure and summary statistics:

```python
Copy code
df.info()
df.describe()
```
Understanding the data's shape, types, and summary statistics is crucial for preprocessing and model selection.

## Step 4: Class Distribution
Let's examine the distribution of fraudulent and non-fraudulent transactions:

```python
Copy code
class_names = {0: 'Not Fraud', 1: 'Fraud'}
print(df.Class.value_counts().rename(index=class_names))
This will help us understand the imbalance in the dataset, which is common in fraud detection scenarios.
```

## Step 5: Data Visualization
We visualize the transaction features to identify patterns and anomalies:

```python
Copy code
fig = plt.figure(figsize=(15, 12))
for i in range(1, 29):
    plt.subplot(5, 6, i)
    plt.plot(df[f'V{i}'])
plt.subplot(5, 6, 29)
plt.plot(df['Amount'])
plt.show()
```
Visualizing features helps in understanding their distribution and identifying potential preprocessing steps.

## Step 6: Splitting the Data
We split the data into training and testing sets:

```python
Copy code
from sklearn.model_selection import train_test_split

feature_names = df.iloc[:, 1:30].columns
target = df.iloc[:, 30].name
data_features = df[feature_names]
data_target = df[target]

X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, train_size=0.70, test_size=0.30, random_state=1)
print(f"Length of X_train is: {len(X_train)}")
print(f"Length of X_test is: {len(X_test)}")
print(f"Length of y_train is: {len(y_train)}")
print(f"Length of y_test is: {len(y_test)}")
```
This ensures that our model is trained on one portion of the data and tested on another, unseen portion.

## Step 7: Model Training
We train a logistic regression model to classify transactions:

```python
Copy code
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

model = LogisticRegression()
model.fit(X_train, y_train.values.ravel())

pred = model.predict(X_test)
Logistic regression is a good starting point for binary classification problems like this one.
```

## Step 8: Evaluating the Model
We evaluate the model's performance using a confusion matrix:

```python
Copy code
class_names = ['not_fraud', 'fraud']
matrix = confusion_matrix(y_test, pred)
dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)
sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues", fmt='g')
plt.title("Confusion Matrix")
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.tight_layout()
plt.show()
```
The confusion matrix provides insight into the number of correctly and incorrectly classified transactions.

## Step 9: Performance Metrics
Finally, we calculate the F1 score and recall, which are crucial for imbalanced datasets:

```python
Copy code
from sklearn.metrics import f1_score, recall_score

f1 = round(f1_score(y_test, pred), 2)
recall = round(recall_score(y_test, pred), 2)
print(f"Sensitivity/Recall for Logistic Regression Model: {recall}")
print(f"F1 Score for Logistic Regression Model: {f1}")
```
The recall (sensitivity) indicates the model's ability to detect fraudulent transactions, while the F1 score balances precision and recall.

## Conclusion
In this blog, we've demonstrated a basic approach to detecting credit card fraud using Python. We've covered data loading, exploration, visualization, model training, and evaluation. This workflow serves as a foundation for more advanced fraud detection techniques, which can include complex algorithms, feature engineering, and real-time detection systems. Stay tuned for more on enhancing fraud detection models!
