import pandas as pd
import openpyxl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dim = pd.read_excel('C:\\Users\\Rohit\\Downloads\\Lab Session1 Data.xlsx').iloc[:, :5]

A = dim.iloc[:, 1:4]
C = dim.iloc[:, 4]

vector_dimension = A.shape[1]
vector_space = A.shape[0]

rank_A = np.linalg.matrix_rank(A)
psuedo_inverse_A = np.linalg.pinv(A)

A_inv_C = np.dot(psuedo_inverse_A, C)

n=C.shape[0]
print(n)
for i in range(n):
    if C[i]>200:
        dim.loc[i, "category"] = "rich"
    else:
        dim.loc[i, "category"] = "poor"
        
df1 = pd.read_excel('C:\\Users\\Rohit\\Downloads\\Lab Session1 Data.xlsx', sheet_name=1)

mean_D = df1['Price'].mean()
variance_D = df1['Price'].var()
wednesday_df = df1[df1['Day'] == 'Wed']
wednesday_mean = wednesday_df['Price'].mean()
population_mean = df1['Price'].mean()

April_df = df1[df1['Month'] == 'Apr']
April_mean = April_df['Price'].mean()
population_mean = df1['Price'].mean()

l2 = list(map(lambda v: v < 0, df1['Chg%']))
l2_false = [value for value in l2 if value is False]
probability = (len(l2_false) / len(l2))*100

l3 = list(map(lambda v: v > 0, wednesday_df['Chg%']))
l3_True = [value for value in l3 if value is True]
probability_wed = (len(l3_True) / len(l3))*100
conditional_prob = probability_wed / wednesday_df.shape[0]


sns.scatterplot(x='Day', y='Chg%', data=df1)


print(dim)

print(A)
print(C)

print(vector_dimension)
print(vector_space)

print(rank_A)
print(psuedo_inverse_A)

print(A_inv_C)

print(dim)

print(df1)

print('Mean:', mean_D)
print('Variance:', variance_D)

print('Wednesday Mean:', wednesday_mean)
print('Population Mean:', population_mean)

print('April Mean:', April_mean)
print('Population Mean:', population_mean)

print(f'Probability: {probability}%')

print(f'profits on wednesday: {probability_wed}%')
print(f'conditional probability: {conditional_prob}%')

plt.show()
