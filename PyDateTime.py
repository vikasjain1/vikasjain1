
import datetime as dt
import time as tm
import pandas as pd
import numpy as np

print("The capital of {province} is {capital}".format(province="Ontario",capital="Toronto"))

s = "Python"
print('\n1.............' + s.center(20)) 

dtnow = dt.datetime.fromtimestamp(tm.time())
print('\n2..............' , dtnow)

dates = pd.date_range("20210101", periods=6)
print('\n3..............' , dates[0])

df = pd.DataFrame(np.random.randn(6, 4), index=dates , columns=list("ABCD"))
print('\n4..............' , df)