import json
import os
from features_extraction import *

path='dataset1/Training'
window_size=100


def get_acc_data_from_json(path,window_size):
  # Read JSON file
  with open(path, 'r') as f:
    data = json.load(f)

  # Get values from JSON
  rot_acc_z = data['rot_acc_z']
  anomalies = data['anomalies']



  # Create sub-lists using a moving sliding window
  step_size = window_size // 2
  #sub_lists = [rot_acc_z[i:i+window_size] for i in range(len(rot_acc_z) - window_size + 1)]
  sub_lists = [rot_acc_z[i:i+window_size] for i in range(0, len(rot_acc_z) - window_size + 1, step_size)]

  # Create list of anomaly types corresponding to each sub-list
  anomaly_types = []
  for sub_list in sub_lists:
      anomaly_type = None
      for anomaly in anomalies:
          if anomaly['start']+window_size//2 <= sub_lists.index(sub_list) <= anomaly['end']-window_size//2:
              anomaly_type = anomaly['type']
              break
      anomaly_types.append(anomaly_type)

  # Create DataFrame
  result_list = [extract_features(item,True,True,True) for item in sub_lists]
  df = pd.DataFrame({'sub_list': result_list, 'anomaly_type': anomaly_types})

  # Drop rows where anomaly_type is None
  df = df.dropna(subset=['anomaly_type'])
  #print("This is the dataframe from a single json file")
  #print(df)
  return df

def load_json_files(directory):
    all_data = pd.DataFrame()
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            path_data= get_acc_data_from_json(file_path,window_size)
            all_data = pd.concat([all_data, path_data])
    # Reset the index of the final DataFrame
    all_data = all_data.reset_index(drop=True)
    #print("This is the dataframe from the directory")
    #print(all_data)
    return all_data

def pre_process_data(d):
  d['new'] = d['sub_list'].apply(lambda x: extract_features(x))
  d = d.drop('sub_list', axis=1)
  d = d.rename(columns={'new': 'sub_list'})
  # Create a mapping from the textual labels to integer numbers
  #label_map = {label: i for i, label in enumerate(d['anomaly_type'].unique())}

  # Map the textual labels to integer numbers
  #d['anomaly_type'] = d['anomaly_type'].map(label_map)

  # Print the updated DataFrame
  #print("This is the dataframe after preprocessing")
  #print(d)
  return(d)

def Windows():

    # Load the dataset as a pandas DataFrame
    accelerometer_data_= load_json_files(path)
    accelerometer_data = pre_process_data(accelerometer_data_)

    return accelerometer_data