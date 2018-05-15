import numpy as np 
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv("skyserver.csv")
# print(data)

train, test = train_test_split(data, test_size=0.2)
print(len(train))
print(len(test))
print(test)