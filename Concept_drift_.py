import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from river import compose, linear_model, metrics, preprocessing, stream

# Simulate concept drift (example)
from sklearn.preprocessing import LabelEncoder

from Preprocessing import *
from read_dataset33 import *


def simulate_concept_drift(data, drift_position):
    # Modify data based on drift position
    return data  # Implement your drift simulation logic here

# Load your accelerometer data into a DataFrame (replace this with your data loading code)
main_folder = 'dataset3'  # Replace with your folder path

#dataset2
data=get_data3(main_folder)
#df = read_data1()
# Create a DataFrame from the collected data
df = pd.DataFrame(data)

label_encoder = LabelEncoder()
df['Category'] = label_encoder.fit_transform(df['Category'])
# Get the unique numerical labels
unique_labels = df['Category'].unique()

df = get_features(df)
X,y = split(df)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
# Initialize metrics
baseline_metrics = metrics.Accuracy()
incremental_metrics = metrics.Accuracy()

# Initialize models
baseline_model = compose.Pipeline(preprocessing.StandardScaler(), linear_model.LogisticRegression())
incremental_model = compose.Pipeline(preprocessing.StandardScaler(), linear_model.SoftmaxRegression)

# Create a data stream from the testing set
data_stream = stream.iter_pandas(X_test, y_test)

# Experiment parameters
drift_positions = [1000, 2000, 3000]  # Example drift positions
drift_window_size = 100  # Example window size to monitor drift
drift_detected = False

# Iterate through the data stream
for i, (X, y) in enumerate(data_stream):
    # Check for concept drift
    if i in drift_positions:
        X = simulate_concept_drift(X, i)  # Simulate concept drift

    # Update the baseline model and metrics
    y_pred_baseline = baseline_model.predict(X)
    baseline_metrics = baseline_metrics.update(y, y_pred_baseline)

    # Update the incremental model and metrics
    y_pred_incremental = incremental_model.predict_one(X)
    incremental_metrics = incremental_metrics.update(y, y_pred_incremental)

    # Detect concept drift (example: drift detected if window average accuracy drops below a threshold)
    if i % drift_window_size == 0:
        accuracy_baseline = baseline_metrics['accuracy_score']
        accuracy_incremental = incremental_metrics['accuracy_score']

        if abs(accuracy_baseline - accuracy_incremental) > 0.05:  # Adjust the threshold as needed
            drift_detected = True

    # Handle concept drift (retrain incremental model if drift is detected)
    if drift_detected:
        incremental_model = compose.Pipeline(preprocessing.StandardScaler(), linear_model.LogisticRegression())
        drift_detected = False

# Print results
print("Baseline Metrics:")
print(baseline_metrics)

print("Incremental Metrics:")
print(incremental_metrics)