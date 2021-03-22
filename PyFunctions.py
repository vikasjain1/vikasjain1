def doSomething (a, b):
    return a + b


print(doSomething(7, 4));
print(type(doSomething))
print('----------------------------')

name = 'Dr. Christopher Brooks'
print(name[4:15])
print(name.split(' ')[0])
print(name.split(' ')[2])

print('----------------------------')
name_phone = {'vikas':498, 'rinki':771, 'riya':972, 'ashu':981}
print (name_phone['vikas'])

for name in name_phone:
    print(name)
    
for phone in name_phone.values():
    print (phone)
    
print('---------------------------')
people = ['Dr. Christopher Brooks', 'Dr. Kevyn Collins-Thompson', 'Dr. VG Vinod Vydiswaran', 'Dr. Daniel Romero']


def split_title_and_name(person):
    return person.split()[0] + ' ' + person.split()[-1]


print (map(split_title_and_name, people))
# option 1
for person in people:
    print(split_title_and_name(person) == (lambda person:person.split(' ')[0] + person.split(' ')[2]))

print('---------------------------')


def myfunc(n):
    return lambda a: a * n


doubler = myfunc(2)
print(doubler(4))
