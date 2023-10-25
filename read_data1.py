import os
from features_extraction import *
import json
from IPython.display import display
import pandas as pd
def read_data1():

    # Define a list to store the data from all JSON files
    data_list = []

    # Directory containing the JSON files
    json_dir = 'dataset1/Training/'  # Replace with your directory path

    # Iterate through each JSON file in the directory
    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(json_dir, filename)

            # Read the JSON file
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
            start_no=0
            # Extract the relevant data
            rot_acc_y = data.get('rot_acc_y', [])
            rot_acc_x = data.get('rot_acc_x', [])
            rot_acc_z_full = data.get('rot_acc_z', [])
            anomalies = data.get('anomalies', [])
            speed_data = data.get('speed', [])

            # Iterate through anomalies and speed data
            for anomaly in anomalies:
                start = anomaly.get('start', None)
                end = anomaly.get('end', None)
                type_anomaly = anomaly.get('type', None)

                if start is not None and end is not None and type_anomaly is not None:
                    # Extract speed data corresponding to the anomaly
                    speed = None
                    for speed_entry in speed_data:
                        if speed_entry.get('start', None) == start and speed_entry.get('end', None) == end:
                            speed = speed_entry.get('speed', None)
                            break

                    # Append data to the list
                    rot_acc_y = data.get('rot_acc_y', [])[start:end + 1]
                    rot_acc_x = data.get('rot_acc_x', [])[start:end + 1]
                    rot_acc_z = data.get('rot_acc_z', [])[start:end + 1]
                    if end<len(rot_acc_z_full) and start_no!=start and start_no<len(rot_acc_z_full)-1:
                        rot_acc_z_no=data.get('rot_acc_z', [])[start_no:start]


                    #feature_y = extract_features(rot_acc_y)
                    #feature_x = extract_features(rot_acc_x)
                    #feature_z = extract_features(rot_acc_z)
                    #feature_regular=extract_features(rot_acc_z_no)

                    #data_list.append([feature_y, feature_x, feature_z, type_anomaly, speed,end-start+1])
                    data_list.append([rot_acc_z_no,"Regular Road",0,start-start_no+1])
                    data_list.append([rot_acc_z, type_anomaly, speed, end - start + 1])
                    start_no = end + 1
    # Create a DataFrame from the list
    #df = pd.DataFrame(data_list, columns=['rot_acc_y', 'rot_acc_x', 'rot_acc_z', 'type_anomaly', 'speed','Duration'])
    df = pd.DataFrame(data_list, columns=['acc_z', 'Category', 'speed', 'Duration'])

    # Display the DataFrame
    return(df)

