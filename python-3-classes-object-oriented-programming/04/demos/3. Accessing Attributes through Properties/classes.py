import logging

logging.basicConfig(level=logging.INFO)


class Employee:
    def __init__(self, name, age, position, salary):
        self.name = name
        self.age = age
        self.position = position
        self._salary = 0
        self.salary = salary

    @property
    def salary(self):
        self._salary = round(self._salary, 2)
        logging.info("Someone has accessed the salary attribute.")
        return self._salary

    @salary.setter
    def salary(self, salary):
        if salary < 1000:
            raise ValueError('Minimum wage is $1000.00')
        self._salary = salary

    def increase_salary(self, percent):
        self._salary *= 1 + (percent / 100)

    def __str__(self):
        return (f"{self.name} is {self.age} years old. "
                f"Employee is a {self.position} with the "
                f"salary of ${self._salary:.2f}")

    def __repr__(self):
        return (
            f"Employee("
            f"{repr(self.name)}, {repr(self.age)}, "
            f"{repr(self.position)}, {repr(self.salary)})"
        )


e = Employee(
    "Ji-Soo",
    38,
    "SWE",
    1200
)
e.increase_salary(3)

e1 = Employee(
    "Lauren",
    44,
    "SDET",
    1000
)

e1.increase_salary(2.5)
print(e)
print(e1)
