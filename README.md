# House-price-prediction-in-machine-learning


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv('House price Dataset.csv')
print(df.head())
print(df.describe())
print(df.isnull().sum())
sns.pairplot(df)
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='jet')
plt.title("Correlation Matrix")
plt.show()
X = df[['Area','BHK','Bathroom','Parking']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
print(X_train.isnull().sum())
print(y_train.isnull().sum())
print(X_test.isnull().sum())
X_train = X_train.dropna()
y_train = y_train.dropna()
X_test = X_test.dropna()
print(X_train.describe())
print(y_train.describe())
print(X_test.describe())
y_train = y_train.iloc[:977]  # Keep the first 977 samples of y_train
X_train = X_train.iloc[:1007]  # Keep the first 1007 samples of X_train
# Fitting the model on the training data
model.fit(X_train, y_train)
# Model Evaluation
y_pred = model.predict(X_test)
y_test = y_train.iloc[:249]
X_test = X_test.iloc[:252]
# Mean Squared Error and R-squared for model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)
# Predictions and Visualization
# To visualize the predictions against actual prices, we'll use a scatter plot
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs. Predicted Prices")
plt.show()
# We can also create a residual plot to check the model's performance
residuals = y_test - y_pred
plt.scatter(y_test, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Actual Prices")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()
# Lastly, let's use the trained model to make predictions on new data and visualize the results
new_data = [[5500, 5, 3, 2]]
predicted_price = model.predict(new_data)
print("Predicted Price:", predicted_price[0])
