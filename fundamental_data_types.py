#Unpacking tuples 
us_num_1, in_num_1 = top_pair[0]
print(us_num_1)
#Chocolate chip
print(in_num_1)
#Punjabi

#For loops and tuple unpacking 
for us_cookie, in_cookie in top_pairs:
print(in_cookie)
print(us_cookie)
#Punjabi
#Chocolate Chip
#Fruit Cake Rusk
#Brownies

#Enumeration is used in loops to return the position and the data 
#in that position while looping

for idx, item in enumerate(top_pairs):
    us_cookie, in_cookie = item
    print(idx, us_cookie, in_cookie)

#(0, 'Chocolate Chip', 'Punjabi')
#(1, 'Brownies', 'Fruit Cake Rusk')

#You can make tupels: zip(), enumerate(), or ()


#Set: unique, unordered, mutable, Python's implementation of Set Theory
#from Mathmatics

cookies_eaten_today = ['chocolate chip', 'peanut butter', 'chocolate chip', 'oatmeal cream', 'chocolate chip']
types_of cookies_eaten = set(cookies_eaten_today)
print(types_of_cookies_eaten)
#set(['chocolate chip', 'oatmeal cream', 'peanut butter'])

#Modifyng sets 
#.add() adds single elements 
#.update() merge in another set or list

types_of_cookies_eaten.add('biscotti')
types_of_cookies_eaten.add('chocolate chip')
print(types_of_cookies_eaten)

cookies_hugo_ate = ['chocolate chip', 'anzac']
types_of_cookies_eaten.update(cookies_hugo_ate)
print(types_of_cookies_eaten)
#set(['chocolate chip', 'anzac', 'oatmeal cream', 'peanut butter', 'biscotti'])

#.discard() safely removes an element from the set by values
#.pop() removes and returns an arbitrary element from the set (KeyError when empty)

types_of_cookies_eaten.discard('biscotti')
print(types_of_cookies_eaten)
#set(['chocolate chip', 'anzac', 'oatmeal cream', 'peanut butter'])

types_of_cookies_eaten.pop()
types_of_cookies_eaten.pop()

print(types_of_cookies_eaten)
'chocolate chip'
'anzac'

#.union() set method returns a set of all the names (or)
#.intersection() method identifies overlapping data (and)
cookies_jason_ate = set(['chocolate chip', 'oatmeal cream', 'peanut butter'])
cookies_hugo_ate = set(['chocolate chip', 'anzac'])
cookies_jason_ate.union(cookies_hugo_ate)

#set(['chocolate chip', 'anzac', 'oatmeal cream', 'peanut butter'])

#.difference() method identifies data present in the set on which 
#the method was used the method was used that is not in the arguments (-)
#Target is important!
cookies_jason_ate.difference(cookies_hugo_ate)
#set(['oatmeal cream', 'peanut butter'])

#performing the reverse of this operation 
cookies_hugo_ate.difference(cookies_jason_ate)
#set(['anzac'])

#You can look for overlapping data in sets by using intersection()
cookies_hugo_ate.intersection(cookies_jason_ate)
#set(['chocolate chip'])




