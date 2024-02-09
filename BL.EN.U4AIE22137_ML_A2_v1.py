import math

def vector_dim(dimension):
  vector=[]
  for i in range(0,dimension):
    vector.append(int(input(f"enter {i+1}th element:")))  #in order to take the user input for the dimensions of vectors
  return vector

vector1=vector_dim(int(input("enter the dimension of X vector")))
vector2=vector_dim(int(input("enter the dimension of Y vector")))

def euclidian_vector(vector1,vector2):
    if(len(vector1)!=len(vector2)):   #we cannot calculate if the dimensions are not equal
        print("The dimensions are not equal")
    
    for i in range(0,len(vector1)):  #it runs the loops according to the length of the first vector
        distance = 0
        distance = distance + ((vector2[i]-vector1[i])*(vector2[i]-vector1[i]))  #formula to calculate euclidian distance
    return math.sqrt(distance)

def manhattan_vector(vector1,vector2):
    if(len(vector1)!=len(vector2)):
        print("The dimensions are not equal")
    
    for i in range(len(vector1)):  #it runs the loops according to the length of the first vector
        distance=0
        distance=distance+((abs(vector2[i]-vector1[i])))  #formula to calculate manhattan distance
    return distance


def manhattan_distance(vector3,vector4):
    distance=0
    for i in range(len(vector3)):  # Loop through the dimensions of the vectors
        distance=distance+(abs(vector4[i]-vector3[i])) 
    return distance
x=[[150],[155],[160],[161],[158]]
y=[[50],[55],[60],[59],[65]]
target=['medium','medium','large','large','large']
value=[]  # To store Manhattan distances
distance={}  # To map distances to target labels
for i in range(len(target)):
    manhattan_dist = sum(abs(a - b) for a, b in zip(x[i], y[i]))    #list comprehension(zip) is used to sort values under x and y automatically.
    value.append(manhattan_dist)    
    distance[manhattan_dist] = target[i]
    print(x[i], y[i], target[i], manhattan_dist)

k=int(input("Enter value of K: "))

sorted_result=sorted(distance.items())  #we are sorting the distance dictionary by keys.


def label_encoder(data):
    map={}  # Create an empty dictionary to map unique labels to numerical values
    encoded_data=[]  # Create an empty list to store the encoded labels
    counter=0
    for i in data:      # Loop through each label in the input data
        if i not in map:          # Check if the label is not already in the map dictionary
            map[i]=counter
            counter=counter+1  # Increment the counter for the next unique label
        encoded_data.append(map[i])   # Append the numerical value corresponding to the label to the encoded_data list
    return encoded_data 


def one_hot_encoding(categories, numerical):
    one_hot_encoded = []  # Create an empty list to store the one-hot encoded vectors
    for i in range(len(numerical)):
        encoded = []  # Create an empty list to store the encoded vector for the current label
        for j in range(len(categories)):  # Loop through each category in the list of categories
            if numerical[i] == j:  # Check if the numerical label matches the index of the current category
                encoded.append(1)  # If the label matches, append 1 to the encoded vector
            else:
                encoded.append(0)  # If the label doesnt match, append 0 to the encoded vector
        one_hot_encoded.append(encoded)
    return one_hot_encoded


result_euclidian=euclidian_vector(vector1,vector2)
result_manhattan=manhattan_vector(vector1,vector2)
print("euclidian is: ",result_euclidian)
print("manhattan is: ",result_manhattan)

print("sorted distance: ",sorted_result)
print("Nearest neighbours are: ",k)
for i in range(k):
    print(sorted_result[i])

data = ['short', 'tall', 'very tall', 'short']
encoded_data = label_encoder(data)
print(encoded_data)


size1 = ['small', 'medium', 'large', 'large', 'medium']
numerical = [0, 1, 2, 2, 1]
size = list(set(size1))  

encoded_data = one_hot_encoding(size, numerical)
for encoding in encoded_data:
    print(encoding)