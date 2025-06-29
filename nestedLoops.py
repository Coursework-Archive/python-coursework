companies = [['Google', 'Facebook', 'Apple'],
            ['Warner Bros. Pictures', '20th Century Fox', 'Universal Pictures'],
            ['Whole foods', 'Starbucks', 'Walmart']]

print(companies[0])
#Output: ['Google','Facebook','Apple']
print(companies[0][1])
#Output: Facebook

for x in [1, 2, 3]:
    for y in [4, 5, 6]:
        print(x * y)
'''
Output:
4
5
6
8
10
12
12
15
18
'''

for x in range(len(companies)):
    print(companies[x])
    print(len(companies[x]))   

'''
['Google', 'Facebook', 'Apple']
3
['Warner Bros. Pictures', '20th Century Fox', 'Universal Pictures']
3
['Whole foods', 'Starbucks', 'Walmart']
3
'''

for x in range(len(companies)):  
    print(companies[x])
    for y in range(len(companies[x])):
        print(y)
'''
Output
['Google', 'Facebook', 'Apple']
0
1
2
['Warner Bros. Pictures', '20th Century Fox', 'Universal Pictures']
0
1
2
['Whole foods', 'Starbucks', 'Walmart']
0
1
2
'''

for x in range(len(companies)):  
    print(companies[x])
    for y in range(len(companies[x])):
        print(companies[x][y])

'''
Output
['Google', 'Facebook', 'Apple']
Google
Facebook
Apple
['Warner Bros. Pictures', '20th Century Fox', 'Universal Pictures']
Warner Bros. Pictures
20th Century Fox
Universal Pictures
['Whole foods', 'Starbucks', 'Walmart']
Whole foods
Starbucks
Walmart
'''

for x in range(len(companies)):  
    print(companies[x])
    for y in range(len(companies[x])):
        print(len(companies[x][y]))

'''
['Google', 'Facebook', 'Apple']
6
8
5
['Warner Bros. Pictures', '20th Century Fox', 'Universal Pictures']
21
16
18
['Whole foods', 'Starbucks', 'Walmart']
11
9
7
'''

guys = ['Giraffe', 'Monkey', 'Panda']
snacks = ['Peanuts', 'Bananas', 'Jelly Babies']
for guy in range(len(guys)):
    for snack in range(len(snacks)):
        print('Gave ' + snacks[snack] + ' to ' + guys[guy])

'''
Gave Peanuts to Giraffe
Gave Bananas to Giraffe
Gave Jelly Babies to Giraffe
Gave Peanuts to Monkey
Gave Bananas to Monkey
Gave Jelly Babies to Monkey
Gave Peanuts to Panda
Gave Bananas to Panda
Gave Jelly Babies to Panda
'''

for guy in guys:
    for snack in snacks:
        if (guy == 'Giraffe' and snack == 'Bananas'):
            continue
        elif (guy == 'Monkey'):
            break
        else:
            print('Gave ' + snack + ' to ' + guy)
        
'''
Output:
Gave Peanuts to Giraffe 
Gave Jelly Babies to Giraffe 
Gave Peanuts to Panda 
Gave Bananas to Panda 
Gave Jelly Babies to Panda 
'''

companies = [['Google', 'Facebook', 'Apple'],
            ['Warner Bros. Pictures', '20th Century Fox', 'Universal Pictures'],
            ['Whole foods', 'Starbucks', 'Walmart']]

for x in range(len(companies)):  
    print(companies[x])
    for y in range(len(companies[x])):
        if (y == 1):
            break
        else:
            print(len(companies[x][y]))

'''
Output:
['Google', 'Facebook', 'Apple']
6
'Warner Bros. Pictures', '20th Century Fox', 'Universal Pictures']
21
['Whole foods', 'Starbucks', 'Walmart']
11
'''

for x in range(len(companies)):  
    print(companies[x])
    for y in range(len(companies[x])):
        if (y == 1):
            continue
        else:
            print(len(companies[x][y]))

'''
Output:
['Google', 'Facebook', 'Apple']
6
5
'Warner Bros. Pictures', '20th Century Fox', 'Universal Pictures']
21
18
['Whole foods', 'Starbucks', 'Walmart']
11
7
'''


#This while loop checks that the word entered matches for one iteration 
word = 'snake'
usr_word = input('Type the word ' + word + ':')
while usr_word != word:
    usr_word = input('Try again!: ')
print('Correct!')

#This program will contnue iteration until the correct number is guessed 
secret_number = 23
guess = None #None is a special word in python that represents that something has no value.
while guess != secret_number:
    guess = input('Enter a number: ')
    guess = int(guess) #int() changes the data type to an integer. In this case the data is being changed from a string.
    if guess > secret_number:
        print('Too high.')
    if guess < secret_number:
        print('Too low.')
print('You got it!')


#Using increment as a flow condition 
start_number = 0
end_number = 10

while start_number != end_number:
    start_number += 1
    print(start_number)