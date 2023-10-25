import copy

import joblib

from river import anomaly
from preprocess_data import *
from Generate_Sliding_Windows import *
from river import compose
from river import linear_model
from river import metrics
from river import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report,roc_curve, precision_recall_curve

accelerometer_data=Windows()
acc_anom=copy.deepcopy(accelerometer_data)
# Create a mapping from the textual labels to integer numbers
label_map = {label: i for i, label in enumerate(accelerometer_data['anomaly_type'].unique())}

# Map the textual labels to integer numbers
accelerometer_data['anomaly_type'] = accelerometer_data['anomaly_type'].map(label_map)

# Load the MLP classifier from the file
print("accelerometer_data",accelerometer_data)
print("acc_anom",acc_anom)
y_true=[]
y_pred_=[]
X_new=[]

model = linear_model.SoftmaxRegression()
metric_acc = metrics.Accuracy()
metric_auc = metrics.ROCAUC()
metric_pr = metrics.Precision()
metric_rec = metrics.Recall()

mlp_loaded = joblib.load("mlp_model.joblib")

names = ['val'+str(i) for i in range(window_size)]
# Stream the data from the DataFrame and train the model incrementally
i=0
for x1, y in zip(accelerometer_data.drop(columns='anomaly_type').to_dict(orient='records'), accelerometer_data['anomaly_type']):
    i+=1
    # Create the dictionary using a dictionary comprehension


    dct = {name: value for name, value in zip(names, x1['sub_list'])}
    x=dct

    #y_true.append(y);
    #X_new.append(x1['sub_list'])

    y_pred = model.predict_one(x)
    # metric = metric.update(y, y_pred)
    metric_acc = metric_acc.update(y, y_pred)
    #acc_values.append(metric_acc.get())

    #metric_auc = metric_auc.update(y, y_pred)
    #auc_values.append(metric_auc.get())

    metric_pr = metric_pr.update(y, y_pred)
    #pr_values.append(metric_pr.get())

    metric_rec = metric_rec.update(y, y_pred)
    #rec_values.append(metric_rec.get())
    #metric = metric.update(y, y_pred)
    model = model.learn_one(x, y)

    '''if i%1000==0:
        mlp_loaded = joblib.load("mlp_model.joblib")
        y_pred_ = mlp_loaded.predict(X_new)
        X_updated = X_new
        y_updated = y_pred_
        mlp_loaded.partial_fit(X_updated, y_updated)'''
# Print the metrics scores

# Print the metrics scores

print(f'Accuracy: {metric_acc.get():.4f}')
print(f'AUC: {metric_auc.get():.4f}')
print(f'Precision: {metric_pr.get():.4f}')
print(f'Recall: {metric_rec.get():.4f}')


def split_lists(row):
    return pd.Series(row['sub_list'])

new_columns = acc_anom.apply(split_lists, axis=1)

# Concatenate the new columns with the original DataFrame
result_df = pd.concat([acc_anom[['anomaly_type']], new_columns], axis=1)

result_df.columns = ['anomaly_type']+['feat_z'+str(i) for i in range(24)]
print("after split",result_df)
custom_mapping = {
    'Regular Road': 0,
    'Bache': 1,
    'Bordo': 2,
    'Boyas': 3,
    # Add more categories and corresponding numbers as needed
  }
result_df['anomaly_type'] = result_df['anomaly_type'].map(custom_mapping)
print("after split___",result_df)
y_anom = result_df['anomaly_type']  # Target labels
X_anom = result_df.drop(columns=['anomaly_type'],axis=1)  # Features (all columns except 'target')


y_pred_anom = mlp_loaded.predict(X_anom)

# Evaluate the model's performance
accuracy = accuracy_score(y_anom, y_pred_anom)
classification_rep = classification_report(y_anom, y_pred_anom)
joblib.dump(mlp_loaded, 'mlp_model.joblib')
# Print the evaluation results
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:\n', classification_rep)
# Confusion matrix plot
# calculate the confusion matrix
#cm = confusion_matrix(y_anom, y_pred_anom)
