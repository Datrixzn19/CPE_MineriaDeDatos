import numpy as np


import pandas as pd


# codigo para leer un csv 

df = pd.read_csv("C:/Users/PERSONAL/Desktop/repos/CPE_MineriaDeDatos/Mental_Health_in_Tech_Survey.csv")
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())