rarebirds = {
    'Gold-crested Toucan': {
        'Height (m)': 1.1,
        'Weight (kg)': 35,
        'Color': 'Gold',
        'Endangered': True,
        'Aggressive': True},
    'Pearlescent Kingfisher': {
        'Height (m)': 0.25,
        'Weight (kg)': 0.5,
        'Color': 'White',
        'Endangered': False,
        'Aggressive': False},
    'Four-metre Hummingbird': {
        'Height (m)': 0.6,
        'Weight (kg)': 0.5,
        'Color': 'Blue',
        'Endangered': True,
        'Aggressive': False},
    'Giant Eagle': {
        'Height (m)': 1.5,
        'Weight (kg)': 52,
        'Color': 'Black and White',
        'Endangered': True,
        'Aggressive': True},
    'Ancient Vulture': {
        'Height (m)': 2.1,
        'Weight (kg)': 70,
        'Color': 'Brown',
        'Endangered': False,
        'Aggressive': False}
    }

birdlocation = [
    'In the canopy directly above our heads.', 
    'Between my 6 and 9 o’clock above.', 
    'Between my 9 and 12 o’clock above.', 
    'Between my 12 and 3 o’clock above.', 
    'Between my 3 and 6 o’clock above.', 
    'In a nest on the ground.', 
    'Right behind you.'
    ]

codes = {
    '---':birdlocation[0],
    '--.':birdlocation[1],
    '-..':birdlocation[2],
    '...':birdlocation[3],
    '..-':birdlocation[4],
    '.--':birdlocation[5],
    '-.-':birdlocation[6]
}

actions = ['Back Away', 'Cover our Heads', 'Take a Photograph']

'''
#Prints the value of the Eagle's Aggressive
for k, v in rarebirds.items():
    if (k == 'Giant Eagle'):
        print(v.get('Aggressive'))
    else:
        continue
#Prints cover our heads if a bird is aggressive
for i, j in rarebirds.items():
    print(i)
    if j.get('Aggressive'):
        print(actions[1])
    else:
        continue

#Prints the morse codes for the location of the birds
for m, n in codes.items():
    print(m + " : " + n)

#Adds a key-value of Seen and False to all of the birds on the list
for p, q in rarebirds.items():
    q.update({'Seen':'False'})

encounter = 'True'

#Takes the input of the user and makes it lower case 
sighting = input("What do you see?: ")
print(sighting.lower())

rarebirdsList = list(rarebirds.keys())
print(rarebirdsList)

#checks if the input is on the rare birds list
for i in rarebirdsList:
    if (sighting == i.lower()):
        print('this is one of the birds we are looking for!')
    elif (sighting != i.lower()):
        print("that's not one of the birds we're looking for")
    else:
        break

code = input('Where do you see it?: ')

location = codes[code]

print("So you've seen a", sighting, location, "My goodness.")

for p, q in rarebirds.items():
    print(p)
    if (p.lower() == sighting.lower() and q.get('Aggressive')):
        print(actions[0], actions[1])
        print("We need to photograph the " + sighting + " " + location)
    elif (p.lower() == sighting.lower() and q.get('Endangered')):
        print("It's endangered. " + actions[0])
        print("We need to photograph the " + sighting + " " + location)
    else:
       print("We need to photograph the ultra rare " + sighting + " " + location)

while sighting.lower() != rarebirds.items():
    sighting = input("What do you see?: ")
    if sighting.lower() == rarebird.items


Action Items to complete the code:
1. error check location to prompt only entries of - or .
2. After seeing a bird on the list, make that bird seen 
value true and continue looking for all untill all the birds have a seen value of true 

'''
actions = ['Back Away', 'Cover our Heads', 'Take a Photograph']

rarebirdsList = list(rarebirds.keys())
encounter = 'True'

print(rarebirdsList)

sighting = input("What do you see? ")


while encounter == 'True':
    sighting = input("Please look for the birds on the rare birds list. What do you see? ")
    for i in rarebirdsList:
        if (i.lower() != sighting.lower()):
            continue
        elif (i.lower() == sighting.lower()):
            code = input('Where do you see it?: ')
            location = codes[code]
            for p, q in rarebirds.items():
                print(q)
                if (q.get('Aggressive')):
                    print(actions[0], actions[1])
                    print("We need to photograph the " + sighting + " " + location)
                    break
                elif (q.get('Endangered')):
                    print("It's endangered. " + actions[0])
                    print("We need to photograph the " + sighting + " " + location)
                    break
                else:
                    print("We need to photograph the ultra rare " + sighting + " " + location)
                    break
            encounter = 'False'
            break


# new_word = 'False'

# print(new_word)

# print(new_word == 'False')

# print(new_word == 'True')

# while new_word == 'False':
#     num = int(input("Enter a number: "))

#     if num > 5:
#         new_word = 'True'
#     if num < 5 or num == 5:
#         print("Try again") 