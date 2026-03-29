import numpy as np
import pandas as pd

data = pd.DataFrame({"id": [1,2,3,4,5,6,7,8,9],
                     "area": [1,2,3,1,2,3,1,2,3],
                     "age": [30, 32, np.nan, 23, 27, 44, 67, 23, np.nan],
                     "amount": [102, 121, 343, np.nan, 121, np.nan, 155, 149, 221]})

#definimos un nuevo objeto que solo tenga los valores sin NaN
data1 = data.dropna()

#Aplicamos el promedio a los datos de edad del objeto "data1"
mean = data1[["age","amount"]].mean()


#Finalmente reemplazamos los datos de Edad en la dataset original
data[["age","amount"]] = data[["age","amount"]].fillna(mean)

print(data)