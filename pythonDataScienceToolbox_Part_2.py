"""
Iterator vs. iterables 
Iterable 
-Examples: lists, strings, dictionaries, file connections 
-An object with an associated iter() method
-Applying iter() to an iterable creates an iterator

Iterator
-Produces next value with next()

word = 'Da'
it = iter(word)
next(it)
'D'
next(it)
'a'
next(it)
error

pythonistas = {'hugo': 'brownw-anderson', 'francis': 'castro'}
for key, value in pythonistas.items()
    print(key, value)
    
file = open('file.txt')
it = iter(file)
print(next(it))

Print(next(it))

"""
##Iterating over iterables 
# Create a list of strings: flash
flash = ['jay garrick', 'barry allen', 'wally west', 'bart allen']

# Print each list item in flash using a for loop
for i in flash:
    print(i)


# Create an iterator for flash: superhero
superhero = iter(flash)

# Print each item from the iterator
print(next(superhero))
print(next(superhero))
print(next(superhero))
print(next(superhero))

##Iterating over iterables (2)
# Create an iterator for range(3): small_value
small_value = iter(range(3))

# Print the values in small_value
print(next(small_value))
print(next(small_value))
print(next(small_value))

# Loop over range(3) and print the values
for num in range(3):
    print(num)


# Create an iterator for range(10 ** 100): googol
googol = iter(range(10 ** 100))

# Print the first 5 values from googol
print(next(googol))
print(next(googol))
print(next(googol))
print(next(googol))
print(next(googol))

##Iterators as function arguments 
# Create a range object: values
values = range(10,21)

# Print the range object
print(values)

# Create a list of integers: values_list
values_list = list(values)

# Print values_list
print(values_list)

# Get the sum of values: values_sum
values_sum = sum(values)

# Print values_sum
print(values_sum)


"""
Enumerate is a function that takes any iterable as argument, such as a lsit and returns a special enumerator object

avengers = ['hawkeye', 'iron man', 'thor', 'quicksilver']
e = enumerate(avengers)
print(type(e))

e_list = list(e) ##List of tuples
print(e_list)

enumerate() is also an iterable and we can unpack its elements, it creates a list of tuples 
avengers = ['hawkeye', 'iron man', 'thor', 'quicksilver']
for index, value in enumerate(avengers): ## indexed at zero 
    print(index, value)

for index, value in enumerate(avengers, start=1): #index at 1
    print(index, value)


Using zip() creates a zip object which is an iterator of tupels
avengers = ['hawkeye', 'iron man', 'thor', 'quicksilver']
names = ['barton', 'stark', 'odinson', 'maximoff']
z = zip(avengers, names)
print(type(z))

z_list = list(z)
print(z_list)
[('hawkeye', 'barton'), ('iron man', 'stark'), ('thor', 'odison'), ('quiclksilver', 'maximoff')] #creates a tuple list based on the position of each string in the elements
"""



# Create a list of strings: mutants
mutants = ['charles xavier', 
            'bobby drake', 
            'kurt wagner', 
            'max eisenhardt', 
            'kitty pryde']

# Create a list of tuples: mutant_list
mutant_list = list(enumerate(mutants))

# Print the list of tuples
print(mutant_list)

# Unpack and print the tuple pairs
for index1, value1 in enumerate(mutants):
    print(index1, value1)

# Change the start index
for index2, value2 in enumerate(mutants, start=1):
    print(index2, value2)
    

# Create a list of tuples: mutant_data
mutant_data = list(zip(mutants, aliases, powers))

# Print the list of tuples
print(mutant_data)

# Create a zip object using the three lists: mutant_zip
mutant_zip = zip(mutants, aliases, powers)

# Print the zip object
print(mutant_zip)

# Unpack the zip object and print the tuple values
for value1, value2, value3 in mutant_zip:
    print(value1, value2, value3)

##Using * and zip to "unzip"

# Create a zip object from mutants and powers: z1
z1 = zip(mutants, powers)

# Print the tuples in z1 by unpacking with *
print(*z1)

# Re-create a zip object from mutants and powers: z1
z1 = zip(mutants, powers)

# 'Unzip' the tuples in z1 by unpacking with * and zip(): result1, result2
result1, result2 = zip(*z1)

# Check if unpacked tuples are equivalent to original tuples
print(result1 == mutants)
print(result2 == powers)

"""
Using iterators to load large files into memory
Reasons why to load data in chunks:
There can be too much data to hold in memory 
Solution: load data in chunks!
Pandas function: read_csv()
##Specify the chunk: chunk_size

import pandas as pd 
result = []
for chunk in pd.read_csv('data.csv', chunksize=1000): #object created is an iterable in which each chunk will be a dataframe
    result.append(sum(chunk['x']))
total = sum(result)
print(total)


#iterating over the data
import pandas as pd
total = 0
for chunk in pd.read_csv('data.csv', chunksize=1000):
    total += sum(chunk['x'])
print(total)
"""

# Initialize an empty dictionary: counts_dict
counts_dict = {}

# Iterate over the file chunk by chunk
for chunk in pd.read_csv('tweets.csv', chunksize=10):

    # Iterate over the column in DataFrame
    for entry in chunk['lang']:
        if entry in counts_dict.keys():
            counts_dict[entry] += 1
        else:
            counts_dict[entry] = 1

# Print the populated dictionary
print(counts_dict)

##Extracting information for large amounts of Twitter data 
# Define count_entries()
def count_entries(csv_file, c_size, colname):
    """Return a dictionary with counts of
    occurrences as value for each key."""
    
    # Initialize an empty dictionary: counts_dict
    counts_dict = {}

    # Iterate over the file chunk by chunk
    for chunk in pd.read_csv(csv_file, chunksize=c_size):

        # Iterate over the column in DataFrame
        for entry in chunk[colname]:
            if entry in counts_dict.keys():
                counts_dict[entry] += 1
            else:
                counts_dict[entry] = 1

    # Return counts_dict
    return counts_dict

# Call count_entries(): result_counts
result_counts = count_entries('tweets.csv', 10, 'lang')

# Print result_counts
print(result_counts)


"""
List comprensions:
-Create lists from other lists, DataFrame columns, etc. 
-Single line of code 
-More efficient than using a for loop 


nums = [12, 8, 21, 3, 16]
new_nums = [num + 1 for num in nums]
print(new_nums)


List comprehension with range()
result = [num for num in range(11)]
print(result)

List comprehension require the following
Collapse for loops for building lists into a single line 
Components
-Iterable
-Iterator variable (represent members of iterable)
-Output expression 


pairs_1 = []
for num1 in range(0, 2):
    for num2 in range(6, 8):
        pairs_1.append(num1, num2)
print(pairs_1)

The same can be done with a list comprehension
pairs_2 = [(num1, num2) for num1 in range(0, 2) for num2 in range(6, 8)]
print(pairs_2)

#The tradeoff is readability
You can build list comprehensions over all the objects except the integer object valjean 
"""

# Create list comprehension: squares
squares = [i**2 for i in range(0,10)]

##Nested list comprehension
# Create a 5 x 5 matrix using a list of lists: matrix
matrix = [[col for col in range(0,5)] for row in range(0,5)]
[]
# Print the matrix
for row in matrix:
    print(row)


"""
More advanced list comprehensions
Conditionals in comprehensions
-Conditionals on the iterable 
[num ** 2 for num in range(10) if num % 2 == 0]

-Conditions on the output expression
[num ** 2 if num % 2 == 0 else 0 for num in range(10)]

Dictionary comprehensions 
-Create dictionaries
-Use curly braces {} instead brackets []
pos_neg = {num: -num for num in range(9)}
print(pos_neg)

print(type(pos_neg))
"""

# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

# Create list comprehension: new_fellowship
new_fellowship = [member for member in fellowship if len(member) >= 7]

# Print the new list
print(new_fellowship)

## Using conditionals in comprehensions (2)
# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

# Create list comprehension: new_fellowship
new_fellowship = [member if len(member) >= 7 else "" for member in fellowship]

# Print the new list
print(new_fellowship)


## Dict comprehensions
# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

# Create dict comprehension: new_fellowship
new_fellowship = {member : len(member) for member in fellowship}

# Print the new dictionary
print(new_fellowship)

"""
Generators are related to list comprehensions
Example of a list comprehension
[2 * num for num in range(10)]

Use () instead of []
This creates a generator object
(2 * num for num in range(10))

A generator is like a list comprehension except it does not store the list in memory

List comprehensions -return a list

Generators - return a generator object

Both can be iterated over
printing values from generators
Ex 1
result = (num for num in range(6))
for num in result:
    print(num)

Ex 2
result = (num for num in range(6))
print(list(result))

Ex 3
result = (num for num in range(6))
-Lazy evaluaton
print(next(result))

print(next(result))

this is handy when yu want to generate large amounts of data without storing them in memory
[num for num in range(10**1000000)] # this is a costly code and will bog up servers

even_nums = (num for num in range(10) if num % 2 == 0)
print(list(even_nums))

Generator functions
Produces generator objects when called
Defined like a regular function - def
Yeilds a sequence of values instead of returning a single value

conditionals in generator expressions 
even_nums = (num for num in range(10) if num % 2 == 0)
print(list(even_nums))

Generator functions 
*Produces generator objects when called
*Defined like a regular function - def
*Yields a sequence of values instead of returing a single value 
*Generates a value with yield keyword 

-Generator function 
def num_sequence(n):
    #Generate values from 0 to n 
    i = 0
    while i < n: 
        yeild i 
        i += 1
"""

# Create generator object: result
result = (num for num in range(31))

# Print the first 5 values
print(next(result))
print(next(result))
print(next(result))
print(next(result))
print(next(result))

# Print the rest of the values
for value in result:
    print(value)
    
    
##Changing the output in generator 
# Create a list of strings: lannister
lannister = ['cersei', 'jaime', 'tywin', 'tyrion', 'joffrey']

# Create a generator object: lengths
lengths = (len(person) for person in lannister)

# Iterate over and print the values in lengths
for value in lengths:
    print(value)


##Build a generator 
# Create a list of strings
lannister = ['cersei', 'jaime', 'tywin', 'tyrion', 'joffrey']

# Define generator function get_lengths
def get_lengths(input_list):
    """Generator function that yields the
    length of the strings in input_list."""

    # Yield the length of a string
    for person in input_list:
        yield len(person)

# Print the values generated by get_lengths()
for value in get_lengths(lannister):
    print(value)
    
    
"""
Recap list comptrehension 
-Basic
[output expression for iterator variable in iterable] 
-Advanced 
[output expression + 
conditional on outpt for iterator variavle in iterble + 
conditional on iterable]
"""

##List comprehensions for time-stamped data 
# Extract the created_at column from df: tweet_time
tweet_time = df['created_at']

# Extract the clock time: tweet_clock_time
tweet_clock_time = [entry[11:19] for entry in tweet_time]

# Print the extracted times
print(tweet_clock_time)

##Conditional list comprehension for time-stamped data 
# Extract the created_at column from df: tweet_time
tweet_time = df['created_at']

# Extract the clock time: tweet_clock_time
tweet_clock_time = [entry[11:19] for entry in tweet_time if entry[17:19] == '19']

# Print the extracted times
print(tweet_clock_time)


"""
Case study on the  World bank data, includes 
-Data on world economies for over half a century 
-Indicators 
Population, Electricity consumption, CO2 emissions, Literacy rates, Unemployment, Mortality rates 

zip() - takes an arbitrary number of iterables and returns an iterator of tuples, you can convert them to a list then priint this list. 

Defining a function 
def raise_both(value1, value2):
#Raise value1 to the power of value2 and vice versa
new_value1 = value1 ** value2
new_value2 = value2 ** value1
new_tuple = (new_value1, new_value2)
return new_tuple 

Basic 
[output expression for iterator variable in iterable]

Advanced 
[output expression + 
conditional on output for iterator variable in iterabel + 
conditional on iterable]
"""

# Zip lists: zipped_lists
zipped_lists = zip(feature_names, row_vals)

# Create a dictionary: rs_dict
rs_dict = dict(zipped_lists)

# Print the dictionary
print(rs_dict)

##Writing a function to help you
# Define lists2dict()
def lists2dict(list1, list2):
    """Return a dictionary where list1 provides
    the keys and list2 provides the values."""

    # Zip lists: zipped_lists
    zipped_lists = zip(list1, list2)

    # Create a dictionary: rs_dict
    rs_dict = dict(zipped_lists)

    # Return the dictionary
    return rs_dict

# Call lists2dict: rs_fxn
rs_fxn = lists2dict(feature_names, row_vals)

# Print rs_fxn
print(rs_fxn)

##Using a list comprehension 
# Print the first two lists in row_lists
print(row_lists[0])
print(row_lists[1])

# Turn list of lists into list of dicts: list_of_dicts
list_of_dicts = [lists2dict(feature_names, sublist) for sublist in row_lists]

# Print the first two dictionaries in list_of_dicts
print(list_of_dicts[0])
print(list_of_dicts[1])

##Turning this all into a DataFrame
# Import the pandas package
import pandas as pd

# Turn list of lists into list of dicts: list_of_dicts
list_of_dicts = [lists2dict(feature_names, sublist) for sublist in row_lists]

# Turn list of dicts into a DataFrame: df
df = pd.DataFrame(list_of_dicts)

# Print the head of the DataFrame
print(df.head())


"""
For large dataset we can use an iterator to load it into chunks
We can also use a generator for the large data limit 
#Use a generator to load a file line by line 
#Works on streaming dat!

def num_squence(n):
    #Generate values from 0 to n.
    i = 0 
    while i < n: 
        yield i 
        i += 1 
"""

# Open a connection to the file
with open('world_dev_ind.csv') as file:

    # Skip the column names
    file.readline()

    # Initialize an empty dictionary: counts_dict
    counts_dict = {}

    # Process only the first 1000 rows
    for j in range(1000):

        # Split the current line into a list: line
        line = file.readline().split(',')

        # Get the value for the first column: first_col
        first_col = line[0]

        # If the column value is in the dict, increment its value
        if first_col in counts_dict.keys():
            counts_dict[first_col] += 1

        # Else, add to the dict and set value to 1
        else:
            counts_dict[first_col] = 1

# Print the resulting dictionary
print(counts_dict)


##Writing a generator to load data in chunks (2)
# Define read_large_file()
def read_large_file(file_object):
    """A generator function to read a large file lazily."""

    # Loop indefinitely until the end of the file
    while True:

        # Read a line from the file: data
        data = file_object.readline()

        # Break if this is the end of the file
        if not data:
            break

        # Yield the line of data
        yield data
        
# Open a connection to the file
with open('world_dev_ind.csv') as file:

    # Create a generator object for the file: gen_file
    gen_file = read_large_file(file)

    # Print the first three lines of the file
    print(next(gen_file))
    print(next(gen_file))
    print(next(gen_file))
    
##Writing a generator to load data in chunks (3)
# Initialize an empty dictionary: counts_dict
counts_dict = {}

# Open a connection to the file
with open('world_dev_ind.csv') as file:

    # Iterate over the generator from read_large_file()
    for line in read_large_file(file):

        row = line.split(',')
        first_col = row[0]

        if first_col in counts_dict.keys():
            counts_dict[first_col] += 1
        else:
            counts_dict[first_col] = 1

# Print            
print(counts_dict)

"""
Reading files in chunks 
#Up next:
-read_csv() function and chunk_size argument
Look at specific indicators in specific countries 
Write a function to generalize the tasks
"""

##Writing on iterator to load data in chunks (1)
# Import the pandas package
import pandas as pd

# Initialize reader object: df_reader
df_reader = pd.read_csv('ind_pop.csv', chunksize=10)

# Print two chunks
print(next(df_reader))
print(next(df_reader))


##Writing an iterator to load data in chunks (2)
# Initialize reader object: urb_pop_reader
urb_pop_reader = pd.read_csv('ind_pop_data.csv', chunksize=1000)

# Get the first DataFrame chunk: df_urb_pop
df_urb_pop = next(urb_pop_reader)

# Check out the head of the DataFrame
print(df_urb_pop.head())

# Check out specific country: df_pop_ceb
df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == 'CEB']

# Zip DataFrame columns of interest: pops
pops = zip(df_pop_ceb['Total Population'], df_pop_ceb['Urban population (% of total)'])

# Turn zip object into list: pops_list
pops_list = list(pops)

# Print pops_list
print(pops_list)

##Writing an iterator to load data in chunks (3)
# Code from previous exercise
urb_pop_reader = pd.read_csv('ind_pop_data.csv', chunksize=1000)
df_urb_pop = next(urb_pop_reader)
df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == 'CEB']
pops = zip(df_pop_ceb['Total Population'], 
           df_pop_ceb['Urban population (% of total)'])
pops_list = list(pops)

# Use list comprehension to create new DataFrame column 'Total Urban Population'
df_pop_ceb['Total Urban Population'] = [int(tup[0] * tup[1] * 0.01) for tup in pops_list]

# Plot urban population data
df_pop_ceb.plot(kind='scatter', x='Year', y='Total Urban Population')
plt.show()


##Writing an iterator to load data in chunks (4)
# Initialize reader object: urb_pop_reader
urb_pop_reader = pd.read_csv('ind_pop_data.csv', chunksize=1000)

# Initialize empty DataFrame: data
data = pd.DataFrame()

# Iterate over each DataFrame chunk
for df_urb_pop in urb_pop_reader:

    # Check out specific country: df_pop_ceb
    df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == 'CEB']

    # Zip DataFrame columns of interest: pops
    pops = zip(df_pop_ceb['Total Population'],
                df_pop_ceb['Urban population (% of total)'])

    # Turn zip object into list: pops_list
    pops_list = list(pops)

    # Use list comprehension to create new DataFrame column 'Total Urban Population'
    df_pop_ceb['Total Urban Population'] = [int(tup[0] * tup[1] * 0.01) for tup in pops_list]
    
    # Append DataFrame chunk to data: data
    data = data.append(df_pop_ceb)

# Plot urban population data
data.plot(kind='scatter', x='Year', y='Total Urban Population')
plt.show()


##Writing an iterator to load data in chunks (5)
# Define plot_pop()
def plot_pop(filename, country_code):

    # Initialize reader object: urb_pop_reader
    urb_pop_reader = pd.read_csv(filename, chunksize=1000)

    # Initialize empty DataFrame: data
    data = pd.DataFrame()
    
    # Iterate over each DataFrame chunk
    for df_urb_pop in urb_pop_reader:
        # Check out specific country: df_pop_ceb
        df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == country_code]

        # Zip DataFrame columns of interest: pops
        pops = zip(df_pop_ceb['Total Population'],
                    df_pop_ceb['Urban population (% of total)'])

        # Turn zip object into list: pops_list
        pops_list = list(pops)

        # Use list comprehension to create new DataFrame column 'Total Urban Population'
        df_pop_ceb['Total Urban Population'] = [int(tup[0] * tup[1] * 0.01) for tup in pops_list]
    
        # Append DataFrame chunk to data: data
        data = data.append(df_pop_ceb)

    # Plot urban population data
    data.plot(kind='scatter', x='Year', y='Total Urban Population')
    plt.show()

# Set the filename: fn
fn = 'ind_pop_data.csv'

# Call plot_pop for country code 'CEB'
plot_pop('ind_pop_data.csv', 'CEB')

# Call plot_pop for country code 'ARB'
plot_pop('ind_pop_data.csv', 'ARB')

