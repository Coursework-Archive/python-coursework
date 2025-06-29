def sqrt(x):
	#Returns the square root of a number. 
	try:
        return x ** 5
    except:
        print('x must be an int or float')
        
sqrt(4)
#2.0

sqrt(10.0)
#3.1622776601683

sqrt('hi')
#x must be an int or float



#Raising an error
def sqrt(x):
    #Returns the square root of a number
    if x < 0:
        raise ValueError('x must be non-negative')
    try:
        return x ** 0.5
    except TypeError:
        print('x must be an int or float')