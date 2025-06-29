# ## this function prints the text between the "" ##

# #print("Hello!")


# ## this creates a variable text and takes an input from the user ##

# #text = input("type some code: ")
# #print(text)


# # store data as the program runs
# # types is python are flexible, this means that it can represent text data, numerical data, true or false data. 
# # Types can change and are flexible.
# # integer type is a whole number (example below: age)
# # If you are curious about the type of data you can print the type using type() and this will show you the 
# # variable type 
# # there is no double in pyton there is a float type (example below: money)
# # Boleans are either true or false values; if they are not true they are false, if they are not false they 
# # are true example(isGameOver)   
# # String data this could be a name a message anything text related and is found between these 
# # double quotes "" or single quotes '' 

# # age = 38
# # age = 39

# # money = 10.10

# # print(type(money))


# # isGameOver = False


# # name = "Brittany"
# # name = 'Brittany'


# # This is a way to take a variable name and assign a value to it
# # # Assignment: =
# # age = 38 
# # first_name = "Brittany "
# # health = 100
# # num_lives = 3

# ## Here are Arithmetic Operators:
# # Addition +
# # Sudtraction -
# # Multiplication *
# # Division /
# # Modulus % - this returns the remainder after division
# # Floor // - this divides and then discard the remainder
# # Exponentiation ** - this is used to raise something to the 
# # power of something else
# # In Python the underscore Syntx is used

# # new_age = age + 1   #Assignment
# # age = age + 1       #Takes the last value of age and adds 1 to it
# # age += 1            #This is an example of the above

# # full_name = first_name + "Tamarkin"

# ## Here are Comparison Operators:
# # Compares two items then either returns true of false depending on whether that comparison is true or not
# # Typically you want to compare two values of the same type 
# # == - returns true if they are equal
# # != - returns true if they are not 
# # >= - returns true if they are greater than or equal
# # > - returns true if they are greater than
# # <= - returns true if they are less than or equal 
# # < - returns true if they are less than

# # is_game_over = health <= 0
# # print(is_game_over)

# ## There are Logical which are an extension of Bolean operators 
# # and - both of these have to be true 
# # or - one of the variables is true then it will return true, if both of them are false it will retrn false
# # not is used to negate a Bolean, it takes a true and makes it false and takes a false and makes is true

# # is_game_over = health <= 0 and num_lives <= 0
# # is_not_game_over = not is_game_over

# #List uses [] to store the list of things (numbers, items, etc.)
# #Lists can also be empty.
# #Lists use commas to separate elements in the list. 
# #A list can be indexed; the following list has 2 indices. Indexing is 0 based.  

# # ## This example changes the invetory item "Food" to 'Fruit'
# # inventory = ["Axe", "Food", "Helmet"]

# # food = inventory[1]
# # inventory[1] = "Fruit"

# # ## Multiple list elements can be accessed by specifying a range
# # ## To do this you use a slicing operator ":" ie: inventory[0:2]. 2 in this case is 
# # ## the upper bound and is not included. The first value in the list is the place to start. 

# # first_two = inventory[0:2] 
# # ##Appending an element to the list 

# # inventory.append("Water")

# # ##Removing an element from the list 

# # inventory.remove("Axe")

# # ##Inserting an element anywhere in the list

# # inventory.insert(1, "Towel")

# # print(inventory)


# ##Tuples are similar to list and immutible, tuples cannot change, you cannot add items to tuples and you
# ##cannot remove items from tuples. They are there to store data 
# #Tuples cannot be assigned new values 

# # profile = ("Brittany", 38)
# # name = profile[0]
# # age = profile[1]
# # print(name)
# # print(age)

# ##Range are use to list a group a whole numbers between a start point and an end point 
# ##A step operator can be used with ranges 
# ##These can be used with loops to tell the it to loop a certain number of time 




# # my_range = range(5)  ##returns (0,5)

# # ##Ranges can also be written as a set of numbers. The first number in the list will 
# # ##be the starting point and the second number is the upper bound and the number 
# # ##listed is not included 

# # ##The following is a range 1 to 100 and it can step by 2, meaning it will print every 
# # ##every other number

# # my_range = range(1,101,2)

# # print(my_range)

# ##Dictionary is similar to the list, it can store multiple values; the big difference 
# ##is the dicitonary stores key value pairs. We can think of it as a map, it is called 
# ##hash map in other languages. With a dictionary the position doesn't matter, all 
# ##that matters is the key under the    


# # inventory = {"Axe":1, "Fruit":3, "Knife":2}

# # num_axes = inventory["Axe"]
# # inventory["Axe"] = 2

# # ##If you are searching for an item that doesn't exist in the dictionary. The
# # ##A new key will be appended to the dictionary  

# # inventory["Helmet"] = 1

# # inventory_keys = inventory.keys()
# # inventory_items = inventory.items()
# # inventory_value = inventory.values()
# # print(inventory_keys)
# # print(inventory_items)
# # print(inventory_value)

# # pos = 5
# # key = "r"


# # if key == "r":
# #     pos += 1
# #     print("Player moved right")
# # elif key == "l":
# #     pos -= 1
# #     print("Player moved left")
# # else:
# #     print("Unknown command")

# # ##You can use 'and' and 'or' with if conditions too 

# ##While loop syntax is run with all text written in the indetation
# ##A while loop is used in circumstances when we don't know how 
# ##many times to run. Typically for game loops and user interaction

# # pos = 0 
# # end_pos = 5

# # while pos < end_pos:   
# #     pos += 1
# #     print(pos)

# # ##Second type of loop is the for loop. For loop is another kind of looop, 
# # ##the for loop has a defined start and end 

# # inventory = ["Axe", "knife", "Helmet"]

# # #Prints all of the inventory that are specified in the list
# # #for item in inventory: 
# #  #   print(item)

# # #Prints all the numbers in the range, not including the upper bound
# # for i in range(5):
# #     # print(i)

# # # ##Calculating interest rate after 5 years 

# # # interest_rate = 1.15
# # # money = 1000
# # # after_five_years = (money * interest_rate ** 5) - 500
# # # print(after_five_years)

# # ##using this amount and adding 5 more years at the same interest rate 
# # after_ten_years = (after_five_years * interest_rate ** 5)
# # print(after_ten_years)

# ##Example of if, elif, else booleans 
# choice = input('Choose a direction: Left, Right, or Straight: ')
# if choice == 'Left':
#     print('You bore me')
# elif choice == 'Right':
#     print('This is getting good')
# elif choice == 'Straight':
#     print('You said up. . right?')
# else:
#     print('You have fingers')


##Checking for even and odd numbers
# num = int(input("Enter a number between 50 and 100: "))
# for i in range(num,201):
#     if i % 2 == 0:
#         print('Fizz') 
#     else:
#         print('Buzz')

# #Starting code 
# shopping_cart = ['eggs', 'banana', 'bread', 'butter']
# #index = 5

# print(shopping_cart[5])


##Randomizing a list, impoting a library and assigning a range printing a random element in the list
# import random
# randomNumber = random.randint(0, 6)
# aList = ['hedgehog', 'newt', 'vole', 'stoat', 'owl', 'toad', 'robin']
# print(len(aList))
# randomElement = aList[randomNumber]
# print(randomElement)



###Reference List Functions in springboard notes for explanation
# guests = [] 
# guests.append('Cuthbert')
# # immediateFamily = ['Mum', 'Dad', 'Grandma', 'Abbie']
# # guests.extend(immediateFamily)
# # list_length = len(guests)
# # print(guests)
# # print(list_length)

# # guests[0] = 'Lucy'
# # print(guests)

# # guests.insert(1, 'Julia')
# # print(guests)

# # del guests[5]
# # print(guests)

# # popped_guest = guests.pop(1)
# # print('Sorry ' + popped_guest + ' come to my next one!')
# # print(guests)


# # guests.remove('Lucy')
# # print(guests)
# # people = len(guests)
# # print(people)

# ##Finds min, max, sum

# # tequillaBottles = [16, 13, 21, 30, 5, 2, 11]
# # print('On the worst day, I had only ', min(tequillaBottles), ' bottles.')
# # print('On the best, day, I had ', max(tequillaBottles), ' bottles.')
# # print("I've had a total of ", sum(tequillaBottles), " bottles in the house this week.")

# # #sorts the list from smallest to largest value
# # print(sorted(tequillaBottles))
# # print(tequillaBottles)


# # ##Practice with list manipulation
# # shopping_cart = ['eggs', 'banana', 'bread', 'butter']
# # shopping_cart[1] = 'mango'
# # shopping_cart.append('rice')
# # shopping_cart.append('soup')
# # shopping_cart.extend(['meat', 'beans'])
# # shopping_cart.insert(1,'grapes')
# # shopping_cart.pop(4)
# # shopping_cart.remove('grapes')
# # print(shopping_cart)


# #Syntax for lists
# # companies =['LinkedIn', 'Google', 'Amazon', 'Apple', 'Microsoft', 'YouTube']
 
# # for i in range(1, len(companies), 3):
# #     print(companies[i])


# # firstFive = list(range(1, 6))
# # oddNumbers = list(range(1, 11, 2))  
# # print(firstFive)
# # print(oddNumbers)

# # # list comprehensions, making a list in one line 
# # squares = [num ** 2 for num in range(1, 11)]
# # print(squares)


# # #More list iteration syntax
# # faveFoods = ['mango', 'pineapple', 'oysters'] 
# # for food in faveFoods:
# #     if food == 'mango':
# #         print("I like",food,"because it is sweet")
# #     elif food == 'pineapple':
# #         print("I like",food,"because it is sweet and sour")
# #     else:
# #         print("I like",food,"from the sea")


# # ## Slice listing things from the 4th element to the end
# # shopping_cart = ['eggs', 'banana', 'bread', 'butter','rice', 'soup', 'meat', 'beans']
# # print(shopping_cart[3:])

# # ##Printing this verticle string
# # s = 'to be or not to be'
# # for i in range(0, len(s)):
# #     print(s[i])

# #Tuples use normal brackets
# # axes = (4, 10)
# # print(axes[0])
# # print(axes[1])

# # axis = (4,)
# # print(axis)

# # for i in axes:
# #     print(i)


# ##Creating and calling a function

# # def my_function():
# #     print('My first function')
# # my_function()

# #Creating a function to sum numbers 
# # ##Creating and calling a function
# # def my_function(num, num1):  
# #     total = num + num1
# #     print(total)

# # my_function(15,32)

# # #This function returns the numbers 
# # ##Creating and calling a function
# def adds_numbers(num, num1):  
#     total = num + num1
#     return total

# # adds_numbers(15,32)



# # #Returning and printing functions, global and local variables
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

# # #Defineing a function and adding two default parameters
# # def describeFood(foodCountry = 'Italy', foodName = 'pasta'):
# #     print('My favourite nation for food is', foodCountry)
# #     print('My favourite food from that country is', foodName)
# # describeFood('France', 'cheese')

# #Reassigning variables when calling a function 
# def my_function(a,b):
#     print(a, 'lives in India.')
#     print(b, 'lives in England.')
    
# #my_function('Harry', 'Ajith')
# my_function(b='Harry',a='Ajith')


# ## Using one function to call another that has been defined
# def oddOrEven(num):
#     if (num%2 == 0):
#         return 'even'
#     else:
#         return 'odd'

# def describeNumber(num):
#     if (num < 10):
#         if (oddOrEven(num) == 'even'):
#             return 'the number is even and less than 10'
#         else:
#             return 'the number is odd and less than 10'
#     else:
#         return 'the number is greater than or equal to 10'
# print(describeNumber(7))



# ##Two ways to calculated the mean 

# numbers = [13, 5102, 45, 2301.40, 203, 1502, 3]

# # Starter code - finished in the lists mini-project
# numbers_sum = sum(numbers)

# if len(numbers) >= 1:
#     print(numbers_sum/len(numbers))
# else:
#     print(0)
    
# # Complete the mean function 
# def mean(arg_numbers):
#     numbs_sum = sum(arg_numbers)
    
#     if len(arg_numbers) >= 1:
#         return numbs_sum / len(arg_numbers)
#     else:
#         return 0
    


# # Test code - Do not change anything below this line
# print(mean(numbers))
