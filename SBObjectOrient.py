# # ##Creating and calling a class 

# # # class Cat:
# # #     def __init__(self, name, hairColour, eyeColour, gender):
# # #         self.name = name,
# # #         self.hairColour = hairColour
# # #         self.eyeColour = eyeColour
# # #         self.gender = gender 
    
# # #     def jump(self):
# # #         print('Jumping Joyously!')

# # #     def meow(self):
# # #         print('meowing cutely.')
    
# # #     def eat(self):
# # #         print('Eating abundantly!')
    
# # #     def sleep(self):
# # #         print('Sleeping deeply. . .')
    

# # # Timothy = Cat('Timothy', 'Amber', 'Blue', 'Male')

# # # Timothy.sleep()

# # # Gerald = Cat('Gerald', 'White', 'Brown', 'Male')

# # # Gerald.jump()


# # ##Printing a string four times

# # # def fourTiimes(word):
# # #     return(word * 4)

# # # print(fourTiimes('mole'))

# # ##Function printing versus a function returning becasue they don't store any value 
# # ##in y because the function does not return anything 

# # # def no_Output(number):
# # #     print(number * 4)

# # # y = no_Output(3)
# # # print(y)

# # ##Global versus local variables 

# # giraffes = 5

# # if (giraffes == 5):
# #     income = 100

# # def cashTree():
# #     money = 50000
# #     return money 

# # print(giraffes)
# # print(income)
# # profit = cashTree()
# # print(profit)

# user_profile = {
#     "name" : "Mark",
#     "age" : 16,
#     "skills" : ["Python", "Java"],
#     "address" : "Downtown"
# }

# #Accessing dictionary item by name 
# print(user_profile['name'])

# #Accessing dictionary using get() method
# print(user_profile.get('age'))

# my_dict = {'name':'Jack', 'age': 26}
 
# # update value
# my_dict['age'] = 27
 
# #Output: {'age': 27, 'name': 'Jack'}
# print(my_dict)
 
# # add item
# my_dict['address'] = 'Downtown' 

# create a dictionary
# squares = {1:1, 2:4, 3:9, 4:16, 5:25}  
 
# removes the item 4:16
# print(squares.pop(4))  
# # Outputs: 16
 
# print(squares)
# # Output: {1: 1, 2: 4, 3: 9, 5: 25}

# print(squares.popitem())
# #Output: (5, 25)
 
# print(squares)
# # Output: {1: 1, 2: 4, 3: 9}

# del squares[5]  
 
# # Output: {1:1, 2:4, 3:9, 4:16}
# print(squares)

# # remove all items
# squares.clear()

# print(squares)

# my_dict = {1:2, 2:4, 3:6}

# #Printing all the kys in the dictionary 
# for i in my_dict.keys():
#     print(i)

# """Output:
# 1
# 2
# 3
# """
# #This will also print all the keys in the dictionary 
# for i in my_dict:
#     print(i)

# excitingbirds = {
#   "a":"barn owl",
#   "b":"guinea fowl",
#   "c":"parakeet",
#   "d":"woodpecker",
#   "e":"blue tit"
# }

# #Printing all the key-value pairs
# for k, v in excitingbirds.items():
#     print(k, v)

# """Output:
# a barn owl
# b guinea fowl
# c parakeet 
# d woodpecker
# e blue tit """

# #This method will get only the values
# for i in excitingbirds.values():
#     print(i)

# """Output:
# barn owl
# guinea fowl
# parakeet 
# woodpecker
# blue tit """

user_profile = {
    'name': 'Mark',
    'age': 16,
    'skills': ['Python', 'Java']
}
 
user_list = list(user_profile.items())

#converts dictionariess to a list and converts them to tuples
print(user_list)
'''
Output: [('name', 'Mark'),('age', 16),('skills', ['Python','Java'])
'''

user_profile = {
    'name': 'Mark',
    'age': 16,
    'skills': ['Python', 'Java']
}

#converts dictionariess to a list and converts them to tuples 
user_list = list(user_profile.items())

#converts list to dictionaries and converts them to tuples
user_dict = dict(user_list)


print(user_list)
#Output: [('name', 'Mark'),('age', 16), ('skills', ['Python', 'Java'])]
print(my_dict)
#Output: {'name':'Mark', 'age':16, 'skills:['Python','Java']}



sample = 'red fox is a fire fox'


sample_cap = sample.capitalize()
print(sample_cap)

print(sample.replace('fox','box'))



def check_fifth_char(string):
    if string[4] == 'D':
        return True
    else:
        return False

check_fifth_char(sample)
print(check_fifth_char(sample))

file = open('test_file.txt','w')

#Writing to a new file
file.write('I think programming is pretty cool.')

file.close()

#Reading the file that is just created
file = open('test_file.txt', 'r')

print(file.read())

file.close