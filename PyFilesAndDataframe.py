
import csv
import pandas as pd
import random

print("\nReading..................")

print('\n#Using Pandas to read csv file.................\n')
df = pd.read_csv('../Data/uk-500.csv')
print(df.head(5))

print('\n#2.............................................\n')
print(df.shape)

print('\n3.............................................\n')
size=df.shape[0]
print(size)
df['id']=[random.randint(0,1000) for x in range(size)]
print(df.head(5))

#iloc(row, col)
print('\n4.....First 3 rows............................')
print(df.iloc[0:3])

print('\n5.....All rows of first 2 columns...................')
print(df.iloc[:,0:2])

#iloc(#rows, #cols)
train_data = df.iloc[:int(10)]
test_data = df.iloc[int(10):]
print('\nTrain Data\n', train_data)
print('\nTest Data\n', test_data)

train_data = df.iloc[:10, 1]
test_data = df.iloc[10:20, 1]
print('\nTrain Data\n', train_data)
print('\nTest Data\n', test_data)
