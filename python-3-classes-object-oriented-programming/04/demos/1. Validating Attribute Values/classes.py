import logging

logging.basicConfig(level=logging.INFO)


class Employee:
    def __init__(self, name, age, position, salary):
        self.name = name
        self.age = age
        self.position = position
        self.salary = 0
        self.set_salary(salary)

    def get_salary(self):
        self.salary = round(self.salary, 2)
        logging.info("Someone has accessed the salary attribute.")
        return f"${self.salary:.2f}"

    def set_salary(self, salary):
        if salary < 1000:
            raise ValueError('Minimum wage is $1000.00')
        self.salary = salary

    def increase_salary(self, percent):
        self.salary += self.salary * (percent/100)

    def __str__(self):
        return f"{self.name} is {self.age} years old. Employee is a {self.position} with the salary of ${self.salary:.2f}"

    def __repr__(self):
        return (
            f"Employee("
            f"{repr(self.name)}, {repr(self.age)}, "
            f"{repr(self.position)}, ${self.salary:.2f}"
        )


employee1 = Employee("Ji-Soo", 38, "developer", 1200)
employee2 = Employee("Lauren", 44, "tester", 1000)

print("Before the promotion")
print(employee1)
print(employee2)
print("Year-end reviews were completed and employees received an increase in salary")
employee1.set_salary(2000)
employee2.increase_salary(3)
print(employee1)
print(employee2)
print(employee2.get_salary())
