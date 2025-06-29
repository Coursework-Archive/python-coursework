
string_1 = "Data Science is future!"

string_2 = "Everybody can learn programming!"

string_3 = "You will learn how to program with Python"
#Do not change code above this line
# #prints the length of the string including spaces
# print(len(string_2))
# #Output: 32
# #print the index of the first 'o'; the indices are counted useing the zeroeth #index rule
# print(string_3.index('o'))
# #Output:1

counter = 0
string_3 = "Being a Data Scientist has provided great opportunities for me!"
#To find the number of occurences of a given character in a string, use a for loop and iterate over the given string 
for char in string_3:
    if (char == 'o'):
        counter += 1
 
print(counter)
#Output is 4

string_3 = "Being a Data Scientist has provided great opportunities for me!"
#count is a quicker method to count the identified alpha/numeric character in a string
print(string_3.count('o'))
#Output is 4

# Print the index within string_3 of the first character of the substring 'ovi'
print(string_3.index('ovi'))
#Output: 29
 
# print a count of the occurrences of 'ovi' in string_3
print(string_3.count('ovi'))
#Output: 1

list_1 = [1, 2, 3, 4, 5]
 
#Slicing from index to index
print(list_1[1:3])
 
#Slicing from the beginning of a list up to an index
print(list_1[:3])
 
#Slicing from an index up to the ending of a list
print(list_1[4:])

'''
Output:
[2, 3]
[1, 2, 3]
[5]
'''

string_1 =  "Hello world!"
 
print(string_1[:5])
#Output: Hello

print(string_1[6:])
#Output: world!

print(string_1[1:5])
#Output: ello

string_4 = "Hello World!"
print(string_4.upper())
print(string_4.lower())

'''
Output
HELLO WORLD!

hello world!
'''

string_5 = "Today is a very nice day!"
print(string_5.split(" ")) #split on space

#Output: ['Today', 'is', 'a', 'very', 'nice', 'day!']

string_6 = "Artificial Intelligence is cool!"
print(string_6.startswith("Artificial"))
print(string_6.endswith("nice!"))

'''
Output 
True 
False
'''

