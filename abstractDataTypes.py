# stack = []

# #Adds a new item to the end of the stack - appends the item that is the second parameter to the stack that is the first parameter
# def push(stack, new_item):
#     stack.append(new_item)

# #Removes the last item on the list     
# aList = [9, 2, 5, 1, 6, 0, 10]
# aList.pop()
# print(aList)

# #Output: [9, 2, 5, 1, 6, 0]

# #Pop() list method doesn't behave as pop() stack method out to. In this example you see that it can remove from the middle of the list 
# #The item at index 3 is removed 
# aList = [9, 2, 5, 1, 6, 0, 10]
# aList.pop(3)
# print(aList)

# #Output: [9, 2, 5, 6, 0, 10]

# #This method will not remove anything from the stack if the stack is length 0
# def pop(stack):
#     if len(stack) < 1:
#         return None
#     return stack.pop()



# #This checks if the stack is length 0 and prints empty if it is.
# list_example = []
 
# if len(list_example) == 0:
#     print("Empty")
# else:
#     print("Not empty")

# #Output: Empty 

# #An alternate way of checking for an empty list 
# list_example = []
 
# if len(list_example) == []:
#     print("Empty")

# #Output: Empty 


# list_example = []
# list_example.clear()
# list_example_1 = [1, 2]

# #Empty list evaluate to false as seen with len(stack) == []
# def is_empty(stack):
#     if len(stack) == 0:
#         print("Empty")
#         print(len(stack) == 0)
#         print(len(stack) == [])
#         print(bool(stack))
#         return len(stack) == 0
#     else:
#         print("Not empty")
        

# is_empty(list_example)
# is_empty(list_example_1)


# list_1 = [1, 2, 3, 4, 5]
 
# #print(len(list_1))

# def size(stack):
#     number_of_elements = len(stack)
#     return number_of_elements

# size(list_1)
# print(size(list_1))

# class Stack:
# #Constructor: part of the class that assigns value/initializes, an instance once the class is made
#     def __init__(self):
#         self.stack = []
 
#     def pop(self):
#         if len(self.stack) < 1:
#             return None
#         return self.stack.pop()
 
#     def push(self, item):
#         self.stack.append(item)

# #All function that you have implemented in the previous steps
# def push(stack, new_item):
# 	stack.append(new_item)

# def is_empty(stack):
# 	return stack == []

# def size(stack):
#     return len(stack)

# #Do not change any code above this line.


# def reverse_string(string):
#     #We have defined the stack for you
#     stack = []
    
#     for char in string:
#         push(stack, char) #push to stack

#     new_string = ""

#     while not is_empty(stack): #Check if the stack is empty
#         new_string += "" + stack.pop() #pop the last element

#     return new_string


# #Tests
# #Do not change code below this line
# assert reverse_string("programming") == "gnimmargorp"
# print("Awesome job!")
# print(reverse_string("programming"))

# queue = []

# #method that adds an item to the list, nothong returned 
# def enqueue(queue, new_item):
#     queue.append(new_item)

# #method that removes an item to the list, nothing returned 
# def dequeue(queue):
#     queue.pop(0)


# #method that checks if there are any items in the queue, returns empty boolean of true
# def is_empty(queue):
#     return (len(queue) == 0)

# #size() functions returns the number of elements in a given queue
# def size(queue):
#     return len(queue)


#Functions that you have implemented in the Queue section


def enqueue(queue, new_item):
    queue.append(new_item)
    print(new_item + ' is added to the end of the queue')
    print(queue)

def dequeue(queue):
    #We have added return here, just to return the item that is being removed
    return queue.pop(0)
    
def is_empty(queue):
	return len(queue) == 0

def size(queue):
    return len(queue)

# Do not change code above this line

def hot_potato_simulator(players, turns):
    hot_potato_queue = [] 
    
    for player in players:
        enqueue(hot_potato_queue, player) #Using enqueue function add a player to the queue
    
    while size(hot_potato_queue) > 1: #Using size function check how many elements are there in the queue
        for i in range(turns):
            enqueue(hot_potato_queue, dequeue(hot_potato_queue)) #Enqueue the next-to-last element from the queue to the end

        if len(hot_potato_queue) != 0:
            print(hot_potato_queue[0] + ' got the hot potato')
        dequeue(hot_potato_queue) # Dequeue the HOT POTATO player
    return dequeue(hot_potato_queue) #Dequeue last element from the queue (The winner)


## Do not change code below this line
import random
players = ["Peter", "John", "Luka", "Maria", "Sophia", "Derek"]
random.shuffle(players)
turns = 10
print(hot_potato_simulator(players, turns))
