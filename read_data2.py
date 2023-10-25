import h5py
import pandas as pd

# Open the HDF5 file in read mode
file_path = 'dataset2/data.hdf5'  # Replace with the path to your HDF5 file
hdf5_file = h5py.File(file_path, 'r')

# Initialize lists to store data and anomaly names
data_list = []
anomaly_list = []

# Iterate through groups under 'idx'
for anomaly_name in hdf5_file['idx']:
    # Access datasets within each anomaly group
    anomaly_group = hdf5_file['idx'][anomaly_name]

    # Iterate through datasets within the anomaly group
    for dataset_name in anomaly_group:
        # Extract data from the dataset
        data = anomaly_group[dataset_name][:]  # Assuming (82, 3) shape

        # Append data to the data list
        data_list.append(data[:2])
        print(data[2:].shape)

        # Append the corresponding anomaly name
        anomaly_list.append(anomaly_name)
        print(anomaly_name)

# Create a Pandas DataFrame with data and anomaly columns
df = pd.DataFrame(data_list, columns=['Feature_1'])

# Add the anomaly column
df['Anomaly'] = anomaly_list

# Now, 'df' is a DataFrame with the data and corresponding anomaly names
# You can further process and analyze the data as needed
