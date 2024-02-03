def matrix_multiply(A,B):
    result3 = []
    for i in range(len(A)):  #to iterate through each row of matrix A
        row=[]
        for j in range(len(B[0])):  #to iterate through each column of matrix B
            sum=0
            for k in range(len(B)):
                sum = sum + A[i][k] * B[k][j]   #to multiply row of matrix A and column of matrix B and add them
            row.append(sum)
        result3.append(row)
    return result3

def matrix_power(A,m):
    result3 = A
    for i in range(1,m):  #Intializing from 1 till the value of m.
        result3=matrix_multiply(result3,A)
    return result3

A=[[1,2],[3,4]]
m=3
result3=matrix_power(A,m)
print(result3)
