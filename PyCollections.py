# tuples have parentheses and are immutable

from string import *
import sys

print('VERSION - ' + sys.version)

print('\n1.............................................\n')
for x in range (5):
    print(x)
    
print('\n2.............................................\n')
for x in range (5,10):
    print(x)
    
print('\n3.............................................\n')
for x in range (5,10,2):
    print(x)
    
print("\n4\t LIST [Indexed, Ordered, Changeable, allows Duplicates]....")
aList=[1,2,3,4,3,2,1]
print(aList)
aList[2]=5
print(aList)
print(len(aList))
aList.append(5)
print(aList)
aList.insert(3, 6)
print(aList)

print("\n5\tTupple............")
aTuple=(1,2,3,4,3,2,1)
print(aTuple)
print(aTuple[2])
aTuple = aTuple+aTuple
print(aTuple)

print("\n6\tTupple Sorted............")
print(sorted(aTuple, reverse=True))

print("\n7\tSet............")
aSet={1,2,3,4,3,2,1}
print(aSet)

print("\n8----------------")
x = (1, 2, 'x') 
print(type(x))
print(x)

# lists have brackets and are mutable
y = [1, 2, 'y']
print(type(y)) 
y.append(5.5)
print(y)

print('9----------------------------')
people = ['Dr. Christopher Brooks', 'Dr. Kevyn Collins-Thompson', 'Dr. VG Vinod Vydiswaran', 'Dr. Daniel Romero']


def split_title_and_name(person):
    title = person.split(' ')[0]
    lastName = person.split(' ')[2]
    return title + ' ' + lastName


print(map(split_title_and_name, people))

print('\n10----------------------------')
name_phone = {'vikas':6154985871, 'rinki':771, 'riya':972, 'ashu':981}
print ('Phone #: ' + str(name_phone['vikas']))

print('\n11----------------------------')
list_of_evens = []  # List
for number in range (0, 50):
    if (number % 2 == 0):
        list_of_evens.append(number)
        
print(list_of_evens)

print('\n12--- List Comprehension --')
list_of_odds = [number for number in range (0, 50) if(number % 2) != 0]
print(list_of_odds)

print('\n13----------------------------')


def times_tables():
    lst = []
    for i in range(2, 5):
        for j in range (1, 5):
            lst.append(i * j)
    return lst


print(times_tables())

print('\n14 --- List Comprehension --')
times_tables_compressed = [i * j for i in range (2, 5) for j in range (1, 5)]
        
print(times_tables_compressed)

print('\n15 --- List Comprehension --')
lowercase = 'abc'
digits = '123'

answer = [lc1 + lc2 + dgt1 + dgt2 for lc1 in lowercase for lc2 in lowercase for dgt1 in digits for dgt2 in digits]
print('--------------')
#print(answer)
print('**************')

correct_answer = [a + b + c + d for a in lowercase for b in lowercase for c in digits for d in digits]
correct_answer[:50]  # Display first 50 ids
#print(correct_answer)

print('\n16.....................\n')
x = (10,20,30,40,50)
for var in x:
    print("index "+ str(x.index(var)+1) + ":",var)
