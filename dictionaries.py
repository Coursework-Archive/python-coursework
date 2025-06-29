#Creating and looping through dictionaries 
#Hold data in key/value pairs - tuples 
#Nestable (use a dictionary as the value of a key within a dictionary)
#Iterable 
#Created by dict() or {}
art_galleries = {}
for name, zip_code in galleries:
    art_galleries[name] = zip_code

for name in art_galleries:
    print(name)

#Zwirner David Gallery
#Zwirner & Wirth
#Zito Studio Gallery
#Zetterquist Galleries 
#Zerre Andre Gallery 

#.get() method allows yuo to safely access key without error or exception handling
#If a key is not in the dictionary, .get() returns Non by default or you can supply
#a value to return 

art_galleries.get('Louvre', 'Not Found')
#Not Found

art_galleries.get('Zarre Andre Gallery')
#10011

art_galleries.keys()
#dict_keys(['10021', '10013'.....])
print(art_galleries['10027'])
#{"Paige's Art Gallery": '(212) 531-1577',..}

art_galleries['10027']['Inner City Art Gallery Inc']
#(212) 368-4941

