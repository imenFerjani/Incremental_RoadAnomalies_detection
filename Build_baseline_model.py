import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import seaborn as sns


from read_data1 import *
from preprocess_data import *
df1=read_data1()
df1=Split_features(df1)
# Split the data into features (X) and target labels (y)
y = df1['type_anomaly']  # Target labels
X = df1.drop(columns=['type_anomaly', 'speed','Duration'],axis=1)  # Features (all columns except 'target')

print("this is X", X)
print(y)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and configure the MLP classifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

# Train the MLP classifier on the training data
mlp_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = mlp_classifier.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
joblib.dump(mlp_classifier, 'mlp_model.joblib')
# Print the evaluation results
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:\n', classification_rep)
# Confusion matrix plot
# calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# plot the confusion matrix as a heatmap
sns.heatmap(cm, annot=True, cmap='Purples')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()