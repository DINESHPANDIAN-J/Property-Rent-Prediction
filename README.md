# Rent Prediction Model

## Overview:

Welcome to the Rent Prediction Model repository! This project aims to provide an accurate and efficient solution for predicting rental prices based on various property features. Whether you're a tenant trying to estimate rental costs or a landlord looking for data-driven insights, this model has you covered.

## Model Performance

The XGBoost model achieved a commendable R2 score of 0.797 during evaluation. This metric indicates the proportion of the variance in the target variable that is predictable from the independent variables. A higher R2 score suggests a better fit of the model to the data.

### Evaluation Metrics
- **R2 Score:** 0.797

## Key Features:

- **Machine Learning Model:** Built on the powerful XGBoost algorithm, our model leverages advanced techniques for precise rent predictions.

- **Intuitive Streamlit App:** Experience seamless interaction with our user-friendly Streamlit web application. Easily input property details and receive instant rent predictions.

- **Locality Encoding with Geopy:** The model utilizes the Geopy library to obtain locality information from latitude and longitude coordinates. This information is then encoded to enhance accuracy, ensuring reliable predictions tailored to specific areas.

- **Amenities Consideration:** Factors such as lift availability, gym access, internet connectivity, and more are taken into account for a comprehensive prediction.
  
## Feature Importances:
Here are the feature importances from the XGBoost model:

| Feature             | Importance |
|---------------------|------------|
| property_size       | 0.040874   |
| POOL                | 0.029628   |
| LIFT                | 0.021461   |
| bathroom            | 0.017130   |
| ...                 | ...        |
| locality            | 0.793989   |

The "locality" feature has a significant impact on rent predictions, contributing approximately 79.4% to the model's overall importance.

## How to Use:

1. **Input Property Details:** Select property type, size, age, number of bathrooms, and other features through the intuitive user interface.

2. **Locality-Specific Predictions:** Leverage the model's capability to provide accurate predictions based on specific localities, ensuring relevance to your target area.

3. **Amenities Impact:** Explore how amenities like lift, gym, pool, and more influence predicted rent values.

## Getting Started:

1. **Clone the Repository:**


## Getting Started:

1. **Clone the Repository:**
   ```
   git clone https://https://github.com/DINESHPANDIAN-J/Property-Rent-Prediction.git
   ```

2. **Install Dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App:**
   ```
   streamlit run app.py
   ```

## Contributing:

We welcome contributions from the community! Whether you're interested in enhancing the model's features, improving the user interface, or fixing bugs, your input is valuable. Check out our [Contribution Guidelines](CONTRIBUTING.md) to get started.

## License:

This project is licensed under the [MIT License](LICENSE), allowing you to freely use and modify the code for your purposes.

Feel free to explore the repository and make informed decisions about rental properties using our sophisticated Rent Prediction Model!
