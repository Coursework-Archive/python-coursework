# class Employee

class Employee:
    def __init__(self):
        self.name = "Ji-Soo"
        self.age = 38
        self.position = "developer"
        self.salary = 1200


e = Employee()
print(e.name + " is an employee at this company. She is " + str(e.age) + " years old and is a " + e.position
      + ". Her salary is $" + str(e.salary) + " per day.")

print(e.__class__)
print(e.__dict__)
