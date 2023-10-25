from read_data1 import *
from Preprocessing import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from read_dataset33 import *

main_folder = 'dataset3'

data=get_data3(main_folder)
df = pd.DataFrame(data)
#df = read_data1()


# Assuming you have a DataFrame named df with columns 'acc_z' and 'anomaly_type'

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have a DataFrame named df with columns 'acc_z' and 'anomaly_type'

# Set a custom color palette
custom_palette = sns.color_palette("Set2")

# Distribution of anomaly types
anomaly_type_counts = df['Category'].value_counts()
print(anomaly_type_counts)

# Create a bar plot for the distribution of anomaly types
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")
sns.barplot(x=anomaly_type_counts.index, y=anomaly_type_counts.values, palette=custom_palette)
plt.title('Distribution of Anomaly Types', fontsize=16)
plt.xlabel('Anomaly Type', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Additional summary statistics for 'acc_z' column
acc_z_lengths = df['acc_z'].apply(lambda x: len(x))

# Create a box plot for 'acc_z' list lengths
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
sns.boxplot(x=acc_z_lengths, palette=custom_palette)
plt.title('Box Plot of z-accelerometer List Lengths', fontsize=16)
plt.xlabel('Length', fontsize=14)
plt.xticks(fontsize=12)
plt.show()

# Create a histogram for 'acc_z' list lengths
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
sns.histplot(x=acc_z_lengths, bins=20, kde=True, color='skyblue')
plt.title('Histogram of z-accelerometer signal Lengths', fontsize=16)
plt.xlabel('Length', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
