import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR

# Load your updated dataset
df = pd.read_csv('/content/datatrain11.csv')

# Assume 'rent' is your target variable, and other columns are features
X = df.drop('rent', axis=1)
y = df['rent']

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Best hyperparameters from the grid search
best_hyperparameters = {'colsample_bytree': 0.9, 'learning_rate': 0.2, 'max_depth': 5, 'n_estimators': 200, 'subsample': 0.9}

# Create the XGBoost model with the best hyperparameters
xgb_model = XGBRegressor(**best_hyperparameters)

# Train the model on the training set
xgb_model.fit(X_train, y_train)

# Save the trained model using joblib
joblib.dump(xgb_model, 'final_xgboost_model.pkl')

# Evaluate the model on the validation set
y_valid_pred = xgb_model.predict(X_valid)
mse_valid = mean_squared_error(y_valid, y_valid_pred)
r2_valid = r2_score(y_valid, y_valid_pred)

# Print the mean squared error on the validation set
print(f'Mean Squared Error on Validation Set: {mse_valid}')
print(f'R-squared on Validation Set: {r2_valid}')

# SAVING MODEL.
# Load  test dataset
test_df = pd.read_csv('/content/datatest11.csv')

# Load the trained XGBoost model
xgb_model = joblib.load('final_xgboost_model.pkl')
# Predict the rent for the test data
y_test_pred = xgb_model.predict(test_df)

# Add the predicted rent values to the test DataFrame
test_df['predicted_rent'] = y_test_pred

# Save the DataFrame with predicted rent values to a new CSV file
test_df.to_csv('predicted_rent_results.csv', index=False)