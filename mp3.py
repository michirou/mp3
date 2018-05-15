import numpy as np 
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv("skyserver.csv")

train, validation, test = np.split(data.sample(frac=1), [int(.7*len(data)), int(.8*len(data))])
print("train ->", len(train))
print("test->", len(test))
print("validation->", len(validation))
# print(test)