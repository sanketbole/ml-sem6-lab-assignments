import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset (assuming the CSV file is in the same directory as this script)
data = pd.read_csv('housing.csv')

# Display the dataframe
print(data)

# Data shape
print(data.shape)

# Data info
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Drop rows with missing values
data.dropna(inplace=True)

# Check for missing values again
print(data.isnull().sum())

# Reset index
data.reset_index(inplace=True, drop=True)

# Display unique values in 'ocean_proximity'
print(data['ocean_proximity'].value_counts())

# Encode 'ocean_proximity'
le = LabelEncoder()
data['ocean_proximity'] = le.fit_transform(data['ocean_proximity'])

# Create new features
data["rooms_per_household"] = data["total_rooms"] / data["households"]
data["bedrooms_per_room"] = data["total_bedrooms"] / data["total_rooms"]
data["population_per_household"] = data["population"] / data["households"]

# Display correlation matrix
print(data.corr())

# Plot histograms
data.hist(bins=50, figsize=(20, 10))
plt.show()

# Plot box plots
data.plot(kind='box', subplots=True, layout=(4, 4), figsize=(15, 7))
plt.show()

# Scatter plots and filtering outliers
x = data.copy()

plt.figure(figsize=(12, 6))
sns.scatterplot(x=x['total_bedrooms'], y=x['median_house_value'])
plt.show()

x = x[x['total_bedrooms'] < 2800]

plt.figure(figsize=(12, 6))
sns.scatterplot(x=x['total_rooms'], y=x['median_house_value'])
plt.show()

x = x[x['total_rooms'] < 15000]

plt.figure(figsize=(12, 6))
sns.scatterplot(x=x['population'], y=x['median_house_value'])
plt.show()

x = x[x['population'] < 6500]

plt.figure(figsize=(12, 6))
sns.scatterplot(x=x['households'], y=x['median_house_value'])
plt.show()

x = x[x['households'] < 2000]

plt.figure(figsize=(12, 6))
sns.scatterplot(x=x['median_income'], y=x['median_house_value'])
plt.show()

x = x[x['median_income'] < 9]

print(x.info())

# Heatmap of correlations
plt.figure(figsize=(13, 13))
sns.heatmap(x.corr(), annot=True)
plt.show()

# Prepare data for modeling
x_data = x.drop(["median_house_value"], axis=1).values
y_data = x["median_house_value"].values

# Polynomial features
feature = PolynomialFeatures(degree=3, include_bias=True, interaction_only=False)
x_data = feature.fit_transform(x_data)

# Standard scaling
scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=42)

# Linear regression model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Model performance
train_score = lr.score(x_train, y_train)
test_score = lr.score(x_test, y_test)
print(f'Train Score: {train_score}')
print(f'Test Score: {test_score}')

# Predictions
y_pred = lr.predict(x_test)
df = pd.DataFrame({"Y_test": y_test, "Y_pred": y_pred})

# Plot predictions vs actual values
plt.figure(figsize=(20, 8))
plt.plot(df[:200])
plt.legend(["Actual", "Predicted"])
plt.show()

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Predicting new values
input_features = pd.DataFrame({
    'longitude': [-121.09],
    'latitude': [39.48],
    'housing_median_age': [25.0],
    'total_rooms': [1665.0],
    'total_bedrooms': [374.0],
    'population': [845.0],
    'households': [330.0],
    'median_income': [1.5603],
    'ocean_proximity': [1],
    'rooms_per_household': [1665.0 / 330.0],
    'bedrooms_per_room': [374.0 / 1665.0],
    'population_per_household': [845.0 / 330.0]
})

# Apply the same transformations to the new input data
input_data_poly = feature.transform(input_features)
input_data_scaled = scaler.transform(input_data_poly)
predicted_value = lr.predict(input_data_scaled)
print("Predicted Median House Value:", predicted_value[0])
