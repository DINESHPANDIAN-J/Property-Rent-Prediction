import pandas as pd
data_train = pd.read_csv('loc_lab.csv')

# # Save the DataFrame to a pickle file
data_train.to_pickle('locality_encoding_map_df.pkl')

# locality_encoding_map_df.to_csv('locality_encoding_map_df.csv')
