from extract_features import *
import numpy as np
def get_features(df):
    # Step 1: Data Preprocessing
    # Convert the 'features' column (lists) to NumPy arrays
    df['acc_z'] = df['acc_z'].apply(extract_all_features)
    df['acc_z'] = df['acc_z'].apply(np.array)
    return df

def split(df):
    # Step 2: Data Splitting
    X = np.array(df['acc_z'].tolist())  # Convert features to a NumPy array
    y = df['Category']
    return X,y