import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR

# Load your updated dataset
df = pd.read_csv('/content/datatrain.csv')

# Assume 'rent' is your target variable, and other columns are features
X = df.drop('rent', axis=1)
y = df['rent']

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a list of models and their hyperparameter grids
models = [
    ('XGBoost', XGBRegressor(), {'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 4, 5], 'n_estimators': [50, 100, 200], 'subsample': [0.8, 0.9, 1.0], 'colsample_bytree': [0.8, 0.9, 1.0]}),
    ('RandomForest', RandomForestRegressor(), {'n_estimators': [50, 100, 200], 'max_depth': [3, 4, 5], 'min_samples_split': [2, 5, 10]}),
    ('LinearRegression', LinearRegression(), {}),
    ('Ridge', Ridge(), {'alpha': [0.1, 1.0, 10.0]}),
    ('Lasso', Lasso(), {'alpha': [0.1, 1.0, 10.0]}),
]

# Create a log file to store metrics
log_file = open('model_metrics_log.txt', 'w')

# Iterate over models
for model_name, model, param_grid in models:
    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Save the best model using joblib
    joblib.dump(best_model, f'best_model_{model_name}.pkl')

    # Evaluate the model on the validation set
    y_valid_pred = best_model.predict(X_valid)
    mse_valid = mean_squared_error(y_valid, y_valid_pred)
    r2_valid = r2_score(y_valid, y_valid_pred)

    # Log the metrics
    log_file.write(f'Model: {model_name}\n')
    log_file.write(f'Best Hyperparameters: {grid_search.best_params_}\n')
    log_file.write(f'Mean Squared Error on Validation Set: {mse_valid}\n')
    log_file.write(f'R-squared on Validation Set: {r2_valid}\n')
    log_file.write('\n')

# Close the log file
log_file.close()
