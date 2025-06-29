# Ask the user if they want to add acronym
while True:
    #if acronym is found in the file, print the definition
    acronym = input("Enter an acronym (or type 'exit' to quit): ")
    if acronym == "exit":
        break
    found = False
    with open("software_acronyms.txt", "r") as file:
        for line in file:
            if line.startswith(acronym):
                print(f"Definition of {acronym}: {line.split(' - ')[1].strip()}")
                found = True
                break
    if not found:
        print(f"{acronym} not found.")
        add_another = input("Do you want to add an acronym? (yes/no) ").strip().lower()
        if add_another == "yes":
            acronym = input("What acronym do you want to add? ")
            definition = input("What is the definition? ")
            with open("software_acronyms.txt", "a") as file:
                file.write(f"{acronym} - {definition}\n")
            print(f"{acronym} has been added to the file.")
        elif add_another == "no":
            print("Goodbye!")
            break
        else:
            print("Please answer 'yes' or 'no'.")