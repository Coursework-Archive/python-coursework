#get example
#art_galleries.get('Louvre', 'Not Found')

#List the keys
#art_galleries.keys()

#Print the value
#print(art_galleries['10027'])

#Get inside nested dictionary
#art_galleries['10027']['Inner Ciy Art Gallery Inc']

#or

#.get()

# Print a list of keys from the boy_names dictionary
print(boy_names.keys())

# Print a list of keys from the boy_names dictionary for the year 2013
print(boy_names[2013].keys())

# Loop over the dictionary
for year in boy_names:
    # Safely print the year and the third ranked name or 'Unknown'
    print(year, boy_names[year].get(3, 'Unknown'))

#Add multiple values into dictionary
#.update()
#create tuple: galleries_11234 = [(.......)]
#art_galleries['11234'].update(galleries_11234) - now a dictionary under the key

#delete a key/value pair
#del art_galleries['11234']

#safely removes a key/value from a dictionary
#galleries_10310 = art_galleries.pop('10310')

# Assign the names_2011 dictionary as the value to the 2011 key of boy_names
boy_names[2011] = names_2011


# Update the 2012 key in the boy_names dictionary
boy_names[2012].update([(1, 'Casey'), (2, 'Aiden')])

# Loop over the years in the boy_names dictionary
for year in boy_names:
    # Sort the data for each year by descending rank and get the lowest one
    lowest_ranked =  sorted(boy_names[year], reverse=True)[0]

    # Safely print the year and the least popular name or 'Not Available'
    print(year, boy_names[year].get(lowest_ranked, 'Not Available'))

# Remove 2011 from femalpope_names and store it: female_names_2011
female_names_2011 = female_names.pop(2011)

# Safely remove 2015 from female_names with an empty dictionary as the default: female_names_2015
female_names_2015 = female_names.pop(2015,{})

# Delete 2012 from female_names
del female_names[2012]

# Print female_names
print(female_names)


#.items() method returns an object we can iterate over

#The following is a way to unpack tuples
for gallery, phone_num in art_galleries.items():
    print(gallery)
    print(phone_num)

#.get() does a lot of work to check for a key
#in operator is much more efficient and cleaner

'11234' in art_galleries
#False

# Iterate over the 2014 nested dictionary
for rank, name in baby_names[2014].items():
    # Print rank and name
    print(rank, name)

# Iterate over the 2012 nested dictionary
for rank, name in baby_names[2012].items():
    # Print rank and name
    print(rank, name)

# Check to see if 2011 is in baby_names
if 2011 in baby_names:
    # Print 'Found 2011'
    print('Found 2011')

# Check to see if rank 1 is in 2012
if 1 in baby_names[2012]:
    # Print 'Found Rank 1 in 2012' if found
    print('Found Rank 1 in 2012')
else:
    # Print 'Rank 1 missing from 2012' if not found
    print('Rank 1 missing from 2012')

# Check to see if Rank 5 is in 2013
if 5 in baby_names[2013]:
    # Print 'Found Rank 5'
    print('Found Rank 5')


#Python cv module
#open() function provides a variable that represents a file, takes a path and a mode
#csv.reader() reads a file object and returns the lines from the file as tuples
#.close() method closes file objects

import csv
csvfile = open('ART_GALLERY.csv', 'r')
for row in csv.reder(csvfile):
    print(row)

#Go from CV file to dictionary
#if data doesn't have a header row, you can pass in column names
# slice notation can be used to skip the header csvfile[1:]
for row in csv.DictReader(csvfile):
    print(row)

# Import the python CSV module
import csv

# Create a python file object in read mode for the baby_names.csv file: csvfile
csvfile = open('baby_names.csv','r')

# Loop over a csv reader on the file object
for row in csv.reader(csvfile):
    # Print each row
    print(row)
    #print(row[4])
    # Add the rank and name to the dictionary
    baby_names[row[5]] = row[3]


# Print the dictionary keys
print(baby_names.keys())

# Import the python CSV module
import csv

# Create a python file object in read mode for the `baby_names.csv` file: csvfile
csvfile = open('baby_names.csv', 'r')

# Loop over a DictReader on the file
for row in csv.DictReader(csvfile):
    # Print each row
    print(row)
    # Add the rank and name to the dictionary: baby_names
    baby_names[row['RANK']] = row['NAME']

# Print the dictionary keys
print(baby_names.keys())# Import the python CSV module
import csv

# Create a python file object in read mode for the `baby_names.csv` file: csvfile
csvfile = open('baby_names.csv', 'r')

# Loop over a DictReader on the file
for row in csv.DictReader(csvfile):
    # Print each row
    print(row)
    # Add the rank and name to the dictionary: baby_names
    baby_names[row['RANK']] = row['NAME']

# Print the dictionary keys
print(baby_names.keys())