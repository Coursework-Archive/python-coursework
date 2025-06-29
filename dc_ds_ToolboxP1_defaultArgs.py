# # Global scope - defined in the main body of a script
# # Local scope - defined inside a function
# # Built-in scope - names in the pre-defined built-ins module
#
# def square(value):
#     #Returns the square of a number
#     new_val = value ** 2
#     return new_val
# square(3)
# #9
#
# new_val = 10
# def square(value):
#     #returns the square of a number
#     global new_val #create new global variable
#     new_val = new_val ** 2
#     return new_val
#
# # Create a string: team
# team = "teen titans"
#
# # Define change_team()
# def change_team():
#     """Change the value of the global variable team."""
#
#     # Use team in global scope
#     global team
#
#     # Change the value of team in global: team
#     team = 'justice league'
# # Print team
# print(team)
#
# # Call change_team()
# change_team()
#
# # Print team
# print(team)

#Returning functions

# Define three_shouts
# def three_shouts(word1, word2, word3):
#     """Returns a tuple of strings
#     concatenated with '!!!'."""
#
#     # Define inner
#     def inner(word):
#         """Returns a string concatenated with '!!!'."""
#         return word + '!!!'
#
#     # Return a tuple of strings
#     return (inner(word1), inner(word2), inner(word3))
#
# # Call three_shouts() and print
# print(three_shouts('a', 'b', 'c'))
#
#
# # Define echo
# def echo(n):
#     """Return the inner_echo function."""
#
#     # Define inner_echo
#     def inner_echo(word1):
#         """Concatenate n copies of word1."""
#         echo_word = word1 * n
#         return echo_word
#
#     # Return inner_echo
#     return inner_echo
#
# # Call echo: twice
# twice = echo(2)
#
# # Call echo: thrice
# thrice = echo(3)
#
# # Call twice() and thrice() then print
# print(twice('hello'), thrice('hello'))
#
# #output: hellohello hellohellohello
#
# # Define echo_shout()
# def echo_shout(word):
#     """Change the value of a nonlocal variable"""
#
#     # Concatenate word with itself: echo_word
#     echo_word = word * 2
#
#     # Print echo_word
#     print(echo_word)
#
#     # Define inner function shout()
#     def shout():
#         """Alter a variable in the enclosing scope"""
#         # Use echo_word in nonlocal scope
#         nonlocal echo_word
#
#         # Change echo_word to echo_word concatenated with '!!!'
#         echo_word = echo_word + '!!!'
#
#     # Call function shout()
#     shout()
#
#     # Print echo_word
#     print(echo_word)
#
#
# # Call function echo_shout() with argument 'hello'
# echo_shout('hello')

#output:
#hellohello
#hellohello!!!

#Flexible arguments
def add_all(*args):
    #Sum all values in *args together

    #initialize sum
    sum_all = 0

    #accumulae the sum
    for num in args:
        sum_all += num

    return sum_all

print(add_all(1))
#output: 1

print(add_all(1, 2))
#output: 3

print(add_all(5, 10, 15, 20))
#output: 50

#Flexible arguments **kwargs - keyword arguments; arguments proceded by identifiers
def print_all(**kwargs):
    #print out key-value pairs in **kwargs

    for key, value in kwargs.items():
        print(key + ': ' + value)

print_all(name="dumbledore", job="headmaster")
#output< job: headmaster
#ouotput< name: dumbledore

# Define shout_echo
def shout_echo(word1, echo=1):
    """Concatenate echo copies of word1 and three
     exclamation marks at the end of the string."""

    # Concatenate echo copies of word1 using *: echo_word
    echo_word = word1 * echo

    # Concatenate '!!!' to echo_word: shout_word
    shout_word = echo_word + '!!!'

    # Return shout_word
    return shout_word

# Call shout_echo() with "Hey": no_echo
no_echo = shout_echo("Hey")

# Call shout_echo() with "Hey" and echo=5: with_echo
with_echo = shout_echo("Hey", 5)

# Print no_echo and with_echo
print(no_echo)
print(with_echo)


# Define shout_echo
def shout_echo(word1, echo=1, intense=False):
    """Concatenate echo copies of word1 and three
    exclamation marks at the end of the string."""

    # Concatenate echo copies of word1 using *: echo_word
    echo_word = word1 * echo

    # Make echo_word uppercase if intense is True
    if intense is True:
        # Make uppercase and concatenate '!!!': echo_word_new
        echo_word_new = echo_word.upper() + '!!!'
    else:
        # Concatenate '!!!' to echo_word: echo_word_new
        echo_word_new = echo_word + '!!!'

    # Return echo_word_new
    return echo_word_new

# Call shout_echo() with "Hey", echo=5 and intense=True: with_big_echo
with_big_echo = shout_echo("Hey", 5, True)

# Call shout_echo() with "Hey" and intense=True: big_no_echo
big_no_echo = shout_echo("Hey", intense=True)

# Print values
print(with_big_echo)
print(big_no_echo)

#output:
#HEYHEYHEYHEYHEY!!!
#HEY!!!

# Define gibberish
def gibberish(*args): #Tuple
    """Concatenate strings in *args together."""

    # Initialize an empty string: hodgepodge
    hodgepodge = ''

    # Concatenate the strings in args
    for word in args:
        hodgepodge += word

    # Return hodgepodge
    return hodgepodge

# Call gibberish() with one string: one_word
one_word = gibberish('luke')

# Call gibberish() with five strings: many_words
many_words = gibberish("luke", "leia", "han", "obi", "darth")

# Print one_word and many_words
print(one_word)
print(many_words)

#output<
#luke
#lukeleiahanobidarth

# Define report_status
def report_status(**kwargs):
    """Print out the status of a movie character."""

    print("\nBEGIN: REPORT\n")

    # Iterate over the key-value pairs of kwargs
    for key, value in kwargs.items():
        # Print out the keys and values, separated by a colon ':'
        print(key + ": " + value)

    print("\nEND REPORT")

# First call to report_status()
report_status(name="luke", affiliation="jedi", status="missing")

# Second call to report_status()
report_status(name="anakin", affiliation="sith lord", status="deceased")

# BEGIN: REPORT
#
# name: luke
# affiliation: jedi
# status: missing
#
# END
# REPORT
#
# BEGIN: REPORT
#
# name: anakin
# affiliation: sith
# lord
# status: deceased
#
# END
# REPORT

# Define count_entries()
def count_entries(df, col_name='lang'):
    """Return a dictionary with counts of
    occurrences as value for each key."""

    # Initialize an empty dictionary: cols_count
    cols_count = {}

    # Extract column from DataFrame: col
    col = df[col_name]

    # Iterate over the column in DataFrame
    for entry in col:

        # If entry is in cols_count, add 1
        if entry in cols_count.keys():
            cols_count[entry] += 1

        # Else add the entry to cols_count, set the value to 1
        else:
            cols_count[entry] = 1

    # Return the cols_count dictionary
    return cols_count


# Call count_entries(): result1
result1 = count_entries(tweets_df)

# Call count_entries(): result2
result2 = count_entries(tweets_df, 'source')

# Print result1 and result2
print(result1)
print(result2)

# {'en': 97, 'et': 1, 'und': 2}
# {'<a href="http://twitter.com" rel="nofollow">Twitter Web Client</a>': 24, '<a href="http://www.facebook.com/twitter" rel="nofollow">Facebook</a>': 1, '<a href="http://twitter.com/download/android" rel="nofollow">Twitter for Android</a>': 26, '<a href="http://twitter.com/download/iphone" rel="nofollow">Twitter for iPhone</a>': 33, '<a href="http://www.twitter.com" rel="nofollow">Twitter for BlackBerry</a>': 2, '<a href="http://www.google.com/" rel="nofollow">Google</a>': 2, '<a href="http://twitter.com/#!/download/ipad" rel="nofollow">Twitter for iPad</a>': 6, '<a href="http://linkis.com" rel="nofollow">Linkis.com</a>': 2, '<a href="http://rutracker.org/forum/viewforum.php?f=93" rel="nofollow">newzlasz</a>': 2, '<a href="http://ifttt.com" rel="nofollow">IFTTT</a>': 1, '<a href="http://www.myplume.com/" rel="nofollow">Plume\xa0for\xa0Android</a>': 1}

# Define count_entries()
def count_entries(df, *args):
    """Return a dictionary with counts of
    occurrences as value for each key."""

    # Initialize an empty dictionary: cols_count
    cols_count = {}

    # Iterate over column names in args
    for col_name in args:

        # Extract column from DataFrame: col
        col = df[col_name]

        # Iterate over the column in DataFrame
        for entry in col:

            # If entry is in cols_count, add 1
            if entry in cols_count.keys():
                cols_count[entry] += 1

            # Else add the entry to cols_count, set the value to 1
            else:
                cols_count[entry] = 1

    # Return the cols_count dictionary
    return cols_count


# Call count_entries(): result1
result1 = count_entries(tweets_df, 'lang')

# Call count_entries(): result2
result2 = count_entries(tweets_df, 'lang', 'source')

# Print result1 and result2
print(result1)
print(result2)

# {'en': 97, 'et': 1, 'und': 2}
# {'en': 97, 'et': 1, 'und': 2, '<a href="http://twitter.com" rel="nofollow">Twitter Web Client</a>': 24, '<a href="http://www.facebook.com/twitter" rel="nofollow">Facebook</a>': 1, '<a href="http://twitter.com/download/android" rel="nofollow">Twitter for Android</a>': 26, '<a href="http://twitter.com/download/iphone" rel="nofollow">Twitter for iPhone</a>': 33, '<a href="http://www.twitter.com" rel="nofollow">Twitter for BlackBerry</a>': 2, '<a href="http://www.google.com/" rel="nofollow">Google</a>': 2, '<a href="http://twitter.com/#!/download/ipad" rel="nofollow">Twitter for iPad</a>': 6, '<a href="http://linkis.com" rel="nofollow">Linkis.com</a>': 2, '<a href="http://rutracker.org/forum/viewforum.php?f=93" rel="nofollow">newzlasz</a>': 2, '<a href="http://ifttt.com" rel="nofollow">IFTTT</a>': 1, '<a href="http://www.myplume.com/" rel="nofollow">Plume\xa0for\xa0Android</a>': 1}