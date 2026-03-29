import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

Salario = pd.read_csv("/Users/mcarthur/Desktop/D_Science_platzi/Pandas/Salary.csv")
Salario.head
Salario.shape()