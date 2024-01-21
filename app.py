import pandas as pd
import joblib
import streamlit as st
import os

# Load the trained XGBoost model
xgb_model = joblib.load(r'pikle_files\final_xgboost_model.pkl')

# Load the one-hot encoding map for localities
locality_encoding_map_df = joblib.load('locality_encoding_map_df.pkl')

# Label mappings
furnishing_encoded_mapping = {'NOT_FURNISHED': 0, 'SEMI_FURNISHED': 1, 'FULLY_FURNISHED': 2}
parking_type_mapping = {'NONE': 1, 'TWO_WHEELER': 0, 'FOUR_WHEELER': 2, 'BOTH': 3}
property_type_mapping = {'RK1': 1, 'BHK1': 2, 'BHK2': 3, 'BHK3': 4, 'BHK4': 5}
facing_label = {'E': 2, 'N': 1, 'S': 3, 'W': 4}
building_type_mapping = {'IF': 1, 'IH': 2, 'AP': 3, 'GC': 4}
watersupply_mapping = {'BOREWELL': 3, 'CORPORATION': 1, 'CORP_BORE': 2}



# Streamlit App
st.set_page_config(
    page_title='Rent Prediction App',
    page_icon='🏠'
)

st.title('Rent Prediction App')


# Input form for the user to provide features
property_type = st.selectbox('Property Type:', list(property_type_mapping.keys()))
property_size = st.number_input('Enter Property Size:', value=0)
property_age = st.number_input('Enter Property Age:', value=0)
bathroom = st.number_input('Enter Number of Bathrooms:', value=0)
balconies = st.number_input('Enter Number of Balconies:', value=0)
cup_board = st.number_input('Enter Number of Cupboards:', value=0)
floor = st.number_input('Enter Number of Floors:', value=0)
building_type = st.selectbox('Select Building Type:', list(building_type_mapping.keys()))
furnishing_encoded = st.selectbox('Select Furnishing Type:', list(furnishing_encoded_mapping.keys()))
facing = st.selectbox('Select Facing:', list(facing_label.keys()))
parking_encoded = st.selectbox('Select Parking:', list(parking_type_mapping.keys()))
water_supply = st.selectbox('Select Water Supply:', list(watersupply_mapping.keys()))

# Locality
selected_locality = st.selectbox('Select Locality:', locality_encoding_map_df['locality'].unique())

# Amenities input components
lift = st.radio('LIFT', ['Yes', 'No'])
gym = st.radio('GYM', ['Yes', 'No'])
internet = st.radio('INTERNET', ['Yes', 'No'])
ac = st.radio('AC', ['Yes', 'No'])
club = st.radio('CLUB', ['Yes', 'No'])
intercom = st.radio('INTERCOM', ['Yes', 'No'])
pool = st.radio('POOL', ['Yes', 'No'])
cpa = st.radio('CPA', ['Yes', 'No'])
fs = st.radio('FS', ['Yes', 'No'])
servant = st.radio('SERVANT', ['Yes', 'No'])
security = st.radio('SECURITY', ['Yes', 'No'])
sc = st.radio('SOLAR CHARGING AVAILABILITY', ['Yes', 'No'])
gp = st.radio('GREEN PARK VIEW', ['Yes', 'No'])
rwh = st.radio('Rain HARVESTING AVAILABILITY', ['Yes', 'No'])
stp = st.radio('SEWAGE TREATMENT PLANT AVAILABILITY', ['Yes', 'No'])
hk = st.radio('HOUSE KEEPING AVAILABILITY', ['Yes', 'No'])
pb = st.radio('POWER BACKUP AVAILABILITY', ['Yes', 'No'])
vp = st.radio('VISITOR PARKING AVAILABILITY', ['Yes', 'No'])

trained_on =pd.read_csv('datatrain11.csv')
trained_on.drop('rent', axis=1, inplace=True) 



# Create a DataFrame with the provided features
input_data = pd.DataFrame({
    'type': [property_type_mapping.get(water_supply, 0)],
    'property_size': [property_size],
    'property_age': [property_age],
    'bathroom': [bathroom],
    'facing': [facing_label.get(facing, 0)],
    'cup_board': [cup_board],
    'floor': [floor],
    'water_supply': [watersupply_mapping.get(water_supply, 0)],
    'building_type': [building_type_mapping.get(building_type, 0)],
    'balconies': [balconies],
    'LIFT': [1 if lift == 'Yes' else 0],
    'GYM': [1 if gym == 'Yes' else 0],
    'INTERNET': [1 if internet == 'Yes' else 0],
    'AC': [1 if ac == 'Yes' else 0],
    'CLUB': [1 if club == 'Yes' else 0],
    'INTERCOM': [1 if intercom == 'Yes' else 0],
    'POOL': [1 if pool == 'Yes' else 0],
    'CPA': [1 if cpa == 'Yes' else 0],
    'FS': [1 if fs == 'Yes' else 0],
    'SERVANT': [1 if servant == 'Yes' else 0],
    'SECURITY': [1 if security == 'Yes' else 0],
    'SC': [1 if sc == 'Yes' else 0],
    'GP': [1 if gp == 'Yes' else 0],
    'RWH': [1 if rwh == 'Yes' else 0],
    'STP': [1 if stp == 'Yes' else 0],
    'HK': [1 if hk == 'Yes' else 0],
    'PB': [1 if pb == 'Yes' else 0],
    'VP': [1 if vp == 'Yes' else 0],
    'furnishing_encoded': [furnishing_encoded_mapping.get(furnishing_encoded, 0)],
    'parking_encoded': [parking_type_mapping.get(parking_encoded, 0)]
})

# Check if the selected locality is in the encoding map
if selected_locality in locality_encoding_map_df['locality'].unique():
    # Get the one-hot encoding for the selected locality
    encoded_locality = locality_encoding_map_df['locality'].unique()

    # Create a DataFrame with one-hot encoding for the locality
    locality_columns = [f'locality_{col}' for col in encoded_locality]
    locality_df = pd.DataFrame(0, index=input_data.index, columns=locality_columns)

    # Set the value for the selected locality to 1
    locality_df.loc[:, f'locality_{selected_locality}'] = 1

    # Concatenate locality DataFrame with the input_data
    input_data = pd.concat([input_data, locality_df], axis=1)
    input_data = input_data[trained_on.columns]
else:
    st.warning(f"The selected locality '{selected_locality}' is not in the training data. Predictions may not be accurate.")


# Button to trigger prediction
if st.button('Predict Rent'):
    # Make prediction using the trained model
    predicted_rent = xgb_model.predict(input_data)

    # Display the predicted rent
    st.success(f'Predicted Rent: {predicted_rent[0]:,.2f}')


