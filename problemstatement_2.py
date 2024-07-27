import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset from a local file
data = pd.read_csv('train.csv')

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Summary statistics
print(data.describe())

# Data types and basic info
print(data.info())

# Example: Filling missing 'Age' values with the median age
data['Age'].fillna(data['Age'].median(), inplace=True)

# Example: Dropping 'Cabin' due to high number of missing values
data.drop(columns=['Cabin'], inplace=True)

# Dropping rows with missing values in 'Embarked'
data.dropna(subset=['Embarked'], inplace=True)

# One-hot encode categorical variables
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# Drop unnecessary columns
data.drop(columns=['Name', 'Ticket', 'PassengerId'], inplace=True)

# Define features (X) and target variable (y)
X = data.drop(columns=['Survived'])
y = data['Survived']

# Split the data into training and testing sets (80-20 ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')

# Generate classification report
class_report = classification_report(y_test, y_pred)
print(f'Classification Report:\n{class_report}')

# Display the coefficients of the model
coefficients = pd.DataFrame(model.coef_[0], index=X.columns, columns=['Coefficient'])
print(coefficients.sort_values(by='Coefficient', ascending=False))
