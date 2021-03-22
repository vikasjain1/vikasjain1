class Person:
    department = 'Health IT'
    
    def setName(self, newName):
        self.name = newName

    def setLocation(self, newLocation):
        self.location = newLocation

        
person = Person()
print(person.department)
person.setName('Vikas Jain')
person.setLocation('Brentwood')
print('{} lives in {}'.format(person.name, person.location))

store1 = [5.0, 6.25, 8.50, 3.40]
store2 = [5.5, 6.25, 8.25, 3.75]
cheapest = map(min, store1, store2)
print(cheapest)

expensive = map(max, store1, store2)
print(expensive)
