import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import openpyxl
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

#Q1
df=pd.read_excel("C:\\Users\\Rohit\\Downloads\\documents\\data1.xlsx")

mean0fgrade = df['radius_mean'].mean()
#mean0fgrade=df[mean0fgrade1].mean()
class1_feat_vectors = np.array([])
class2_feat_vectors = np.array([])
 
# Create a new column in df based on the condition
df['results'] = np.where(df['radius_mean'] > mean0fgrade, 1, 0)
 
# Append values to class1_feat_vectors and class2_feat_vectors based on the new column
class1_feat_vecs = np.append(class1_feat_vectors, df[df['results'] == 1]['radius_mean'].values)
class2_feat_vecs = np.append(class2_feat_vectors, df[df['results'] == 0]['radius_mean'].values)
 
centroid1 = class1_feat_vectors.mean(axis=0)
centroid2 = class2_feat_vectors.mean(axis=0)
 
# Calculate the spread (standard deviation) for each class
spread1 = class1_feat_vectors.std(axis=0)
spread2 = class2_feat_vectors.std(axis=0)
 
# Calculate the distance between the centroids of the two classes
interclass_distance = np.linalg.norm(centroid1 - centroid2)
 
print(f"Centroid of Class 1: {centroid1}, Spread: {spread1}")
print(f"Centroid of Class 2: {centroid2}, Spread: {spread2}")
print(f"Interclass Distance: {interclass_distance}")
result = df['results']


#Q2
mean=np.mean(df['radius_mean'])
df['Result']=np.where(df["radius_mean"]>mean ,0,1)
sns.scatterplot(x='Result', y='radius_mean', data=df)

plt.show()
df.head()

#Q4
grad=df['radius_mean']

X_train,X_test,Y_train,Y_test = train_test_split(grad,result, test_size=0.01, shuffle=True)

grad1=np.reshape(grad,(-1,1))

X_train1=np.reshape(X_train,(-1,1))
Y_train1=np.reshape(Y_train,(-1,1))

X_test1=np.reshape(X_test,(-1,1))
Y_test1=np.reshape(Y_test,(-1,1))

#Q5
knn=KNeighborsClassifier(3)
print(knn.fit(X_train1,Y_train))

#Q6
print(knn.score(X_test1,Y_test))

#Q7
print(knn.predict(X_test1))

#Q8
accuracies = []
for k in range(1, 12):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train1, Y_train)
    accuracy = knn.score(X_test1, Y_test)
    accuracies.append(accuracy)

plt.plot(range(1, 12), accuracies)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. k')
plt.show()


#Q9
train_predictions = knn.predict(X_train1)
test_predictions = knn.predict(X_test1)

train_conf_matrix = confusion_matrix(Y_train, train_predictions)
print("Confusion Matrix for Training Data:")
print(train_conf_matrix)

test_conf_matrix = confusion_matrix(Y_test, test_predictions)
print("Confusion Matrix for Test Data:")
print(test_conf_matrix)

train_precision = precision_score(Y_train, train_predictions, average=None)
print("Precision for Training Data:", train_precision)
test_precision = precision_score(Y_test, test_predictions, average=None)
print("Precision for Test Data:", test_precision)

train_recall = recall_score(Y_train, train_predictions, average=None)
print("Recall for Training Data:", train_recall)
test_recall = recall_score(Y_test, test_predictions, average=None)
print("Recall for Test Data:", test_recall)

train_f1 = f1_score(Y_train, train_predictions, average=None)
print("F1-score for Training Data:", train_f1)
test_f1 = f1_score(Y_test, test_predictions, average=None)
print("F1-score for Test Data:", test_f1)