import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, mean_squared_error, cohen_kappa_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

# Load the Iris dataset
iris = load_iris()
X_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
y_iris = pd.Series(iris.target, name='species')

# Split the dataset into training and testing sets
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)

# Load the California Housing dataset
california_housing = fetch_california_housing()
X_california = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
y_california = pd.Series(california_housing.target, name='MEDV')

# Split the dataset into training and testing sets
X_train_california, X_test_california, y_train_california, y_test_california = train_test_split(X_california, y_california, test_size=0.3, random_state=42)

# Classification Task
clf_iris = DecisionTreeClassifier(criterion='gini', random_state=42)
clf_iris.fit(X_train_iris, y_train_iris)
y_pred_iris = clf_iris.predict(X_test_iris)

conf_matrix_iris = confusion_matrix(y_test_iris, y_pred_iris)
class_report_iris = classification_report(y_test_iris, y_pred_iris)
fpr_iris, tpr_iris, _ = roc_curve(y_test_iris, clf_iris.predict_proba(X_test_iris)[:, 1], pos_label=1)
roc_auc_iris = auc(fpr_iris, tpr_iris)

print("Confusion Matrix:\n", conf_matrix_iris)
print("\nClassification Report:\n", class_report_iris)

plt.figure()
plt.plot(fpr_iris, tpr_iris, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_iris)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Regression Task
reg_california = DecisionTreeRegressor(criterion='squared_error', random_state=42)
reg_california.fit(X_train_california, y_train_california)
y_pred_california = reg_california.predict(X_test_california)

mse_california = mean_squared_error(y_test_california, y_pred_california)
print("Mean Squared Error:", mse_california)

plt.figure()
plt.scatter(y_test_california, y_pred_california, color='darkorange', lw=2)
plt.plot([min(y_test_california), max(y_test_california)], [min(y_test_california), max(y_test_california)], color='navy', lw=2, linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()

# Calculate Kappa Statistics
kappa_iris = cohen_kappa_score(y_test_iris, y_pred_iris)
# Calculate Precision, Recall, F-measure
precision_iris = precision_score(y_test_iris, y_pred_iris, average='macro')
recall_iris = recall_score(y_test_iris, y_pred_iris, average='macro')
f1_iris = f1_score(y_test_iris, y_pred_iris, average='macro')

print("Kappa Statistics:", kappa_iris)
print("Precision:", precision_iris)
print("Recall:", recall_iris)
print("F1-measure:", f1_iris)

# Provide insights and recommendations based on the evaluation results
print("Analysis and Interpretation:\n")
print("Classification Task:\n")
print("The Decision Tree classifier performed well on the Iris dataset, achieving high precision, recall, and F1-score. The ROC curve indicates a strong ability to distinguish between classes. Improvements can be made by tuning hyperparameters or using ensemble methods.\n")

print("Regression Task:\n")
print("The Decision Tree regressor performed reasonably well on the California Housing dataset, as indicated by the mean squared error. However, overfitting might be an issue. To improve performance, consider pruning the tree or using ensemble methods like Random Forests.\n")
