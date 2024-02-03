def counting_pairs(arr1):
    count = 0
    checked = set() #defining a set to add items/values
    for num in arr1:
        difference = 10-num  #the value we get after subtracting will be taken to check if the number is there in the array or not.
        if difference in checked:
            count = count + 1 #if the value of difference is there in the set, it adds 1 to the current count value.
        checked.add(num)  #the count value is joined in the checked set. 
    return count

def range(arr2):
    if len(arr2)<3:
        return "Range determination not possible"
    else:
        return max(arr2) - min(arr2)   #by using max and min function, we can find the maximum and minimum value in the defined array. The difference will give us the range.

def highest_occurence(input_string):
    counter1={}   #created a dictionary to store the number of occurences.
    for letter in input_string:
        if letter in counter1:
            counter1[letter]=counter1[letter]+1 #for every letter in the string, it iterates he counter1 dictionary.
        else:
            counter1[letter]=1  #if there is no repitition, its gonna stay 1.
    find_max=max(counter1,key=counter1.get)  #we use max funtion to find the maximum value in the counter1 which is basically the letter that has occured the most.
    return f"Character : {find_max} has highest frequence of : {counter1[find_max]}"


arr1=[2,7,4,7,1,7,3,6]
result1=counting_pairs(arr1)
print(result1)

arr2 = [5,3,8,1,0,4]
result2=range(arr2)
print(result2)

string1="hippopotamus"
result=highest_occurence(string1)
print(result)
