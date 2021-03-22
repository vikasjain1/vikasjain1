x = (1, 2, 'x')  # tuples have parentheses and are immutable
y = [3, 4, 'y']  # lists

for item in x:
    print(item)

print('\n')

for item in y:
    print(item)

print('\n')

i = 0
while (i < len(x)):
    print(x[i] * 10)
    i = i + 1

print(1 in (1, 2, 3))
print(0 in (1, 2, 3))
