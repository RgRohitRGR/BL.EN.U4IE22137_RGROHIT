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
df['results'] = np.where(df['radius_mean'] > mean0fgrade, 1, 0)
result = df['results']

grad=df['radius_mean']

X_train,X_test,Y_train,Y_test = train_test_split(grad,result, test_size=0.01, shuffle=True)

#grad1=np.reshape(grad,(-1,1))

X_train1=np.reshape(X_train,(-1,1))
Y_train1=np.reshape(Y_train,(-1,1))

X_test1=np.reshape(X_test,(-1,1))
Y_test1=np.reshape(Y_test,(-1,1))

knn=KNeighborsClassifier(3)
#print(knn.fit(X_train1,Y_train))
#print(knn.score(X_test1,Y_test))
#print(knn.predict(X_test1))

accuracies = []
for k in range(1, 12):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train1, Y_train)
    accuracy = knn.score(X_test1, Y_test)
    accuracies.append(accuracy)

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

'''def checkfitting(test_predictions,train_predictions):
    if (train_predictions < 0.80).all():
        return ("underfitting")
    elif(train_predictions > 0.80 and test_predictions > 0.80):
        return ("good fitting")
    elif(test_predictions < 0.80):
        return ("Overfitting")
print(checkfitting(test_predictions,train_predictions))'''

def checkfitting(test_predictions, train_predictions):
    if np.all(train_predictions > 0.80) and np.all(test_predictions > 0.80):
        return "Good Fitting"
    elif np.any(train_predictions < 0.80):
        return "Under Fitting"
    elif np.any(test_predictions < 0.80):
        return "Over Fitting"
    
print(checkfitting(test_predictions, train_predictions))


#Q2
df = pd.read_excel("C:\\Users\\Rohit\\Downloads\\documents\\Lab Session1 Data.xlsx", sheet_name='Purchase data')
df=pd.DataFrame(df)
print(df)
#convert datafram to array
Array=df.to_numpy()
A=Array[:,1:4]
C=Array[:,4:5]
#typecast the elements to float
A=np.float64(A)
#return the pseudoinverse
pinv=np.linalg.pinv(A)
print("pseudo inverse:",pinv)

#calculating the cost of each product
print("Costs of the products:",np.matmul(pinv,C))
pinv2=np.matmul(pinv,C)

#comparing real prices with the model for calculating prices
print("real prices:",C)
predicted=np.matmul(A,pinv2)
print("model of calculating prices:",predicted)

#funtion to calculate MSE
def MSE(C,predicted):
    sum=0
    for i in range(len(C)):
        sub=0
        sub=C[i]-predicted[i]
        sub=sub*sub
        sum+=sub
    return sum/len(C)
print("Mean squared error:",MSE(C,predicted))

#funtion to calculate RMSE
def RMSE(C,predicted):
    sum=0
    for i in range(len(C)):
        sub=0
        sub=C[i]-predicted[i]
        sub=sub*sub
        sum+=sub
    return math.sqrt(sum/len(C))
print("RMSE:",RMSE(C,predicted))

#funtion to calculate MAPE
def MAPE(C,predicted):
    sum=0
    for i in range(len(C)):
        sub=0
        sub=C[i]-predicted[i]
        sub=abs((sub)/C[i])
        sum+=sub
    return sum/len(C)
print("MAPE:",MAPE(C,predicted))

#funtion to calculate R2 score
import math
import numpy as np
def R2(C,predicted):
    rss=0
    for i in range(len(C)):
        err=0
        err=C[i]-predicted[i]
        rss += err*err
    tss=0
    mean=np.mean(predicted)
    for i in range(len(C)):
        err=0
        err=C[i]-mean
        tss += err*err   
    return 1-(rss/tss)
print("R2",R2(C,predicted))

#Q3
import numpy as np
import matplotlib.pyplot as plt

X = np.random.randint(1, 11, size=20)
Y = np.random.randint(1, 11, size=20)

class_labels = ([0] * 10 + [1] * 10)

plt.figure(figsize=(8, 6))
plt.scatter(X, Y, c=class_labels, cmap=plt.cm.bwr, edgecolors='k')
plt.colorbar(ticks=[0, 1], label='Class')
plt.title('Training Data Scatter Plot')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()

#Q4
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

x_values = np.arange(0, 10.1, 0.1)
y_values = np.arange(0, 10.1, 0.1)
test_data = np.array([[x, y] for x in x_values for y in y_values])

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(np.column_stack((X, Y)), class_labels)
predicted_labels = knn.predict(test_data)

plt.figure(figsize=(8, 6))
plt.scatter(test_data[:, 0], test_data[:, 1], c=predicted_labels, cmap=plt.cm.bwr, alpha=0.5)
plt.colorbar(ticks=[0, 1], label='Predicted Class')
plt.title('Test Data Scatter Plot ')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()

#Q5
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

x_values = np.arange(0, 10.1, 0.1)
y_values = np.arange(0, 10.1, 0.1)
test_data = np.array([[x, y] for x in x_values for y in y_values])

k_values = [1, 2]

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(np.column_stack((X, Y)), class_labels)
    predicted_labels = knn.predict(test_data)

    plt.figure(figsize=(8, 6))
    plt.scatter(test_data[:, 0], test_data[:, 1], c=predicted_labels, cmap=plt.cm.bwr, alpha=0.5)
    plt.colorbar(ticks=[0, 1], label='Predicted Class')
    plt.title(f'Test Data Scatter Plot')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

#Q6
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

excel_file = 'C:\\Users\\Rohit\\Downloads\\documents\\data1.xlsx' 
df = pd.read_excel(excel_file)
'''
X = df[['radius_mean', 'perimeter_mean']].values
y = df['area_mean'].values'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)  
knn.fit(X_train, y_train)

predicted_labels = knn.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=predicted_labels, cmap=plt.cm.bwr, alpha=0.5)
plt.colorbar(ticks=np.unique(y), label='Predicted Class')
plt.title('Test Data Scatter Plot with Predicted Class Labels')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()

#Q7
#splitting the data into train and set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5)
#applying KNN of K=3
neigh=KNeighborsClassifier(n_neighbors=3)

#initialzing n_neighbors
parameters={'n_neighbors':sp_randInt(1,100)}
'''
#call randomized Search CV to tune the hyperparameter
clf = RandomizedSearchCV(neigh, param_distributions=parameters,random_state=0)
search=clf.fit(X,y)
print("best K value:",search.best_params_)'''







