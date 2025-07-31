import logging

logging.basicConfig(level=logging.INFO)


class Employee:
    def __init__(self, name, age, position, salary):
        self.name = name
        self.age = age
        self.position = position
        self._salary = 0
        self.set_salary(salary)

    def increase_salary(self, percent):
        self._salary += self._salary * (percent/100)

    def get_salary(self):
        self._salary = round(self._salary, 2)
        logging.info("Someone has accessed the salary attribute.")
        return f"${self._salary:.2f}"

    def set_salary(self, salary):
        if salary < 1000:
            raise ValueError('Minimum wage is $1000.00')
        self._salary = salary

    def __str__(self):
        return (f"{self.name} is {self.age} years old. Employee is a "
                f"{self.position} with the salary of ${self._salary:.2f}")

    def __repr__(self):
        return (
            f"Employee("
            f"{repr(self.name)}, {repr(self.age)}, "
            f"{repr(self.position)}, {repr(self._salary)})"
        )


employee1 = Employee("Ji-Soo", 38, "developer", 1200)
employee2 = Employee("Lauren", 44, "tester", 1000)

employee1.set_salary(2000)
print(employee1.get_salary())
print(employee2.get_salary())
