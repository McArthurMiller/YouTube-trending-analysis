import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Paso 1 Carga del dataset
Salary = pd.read_csv("/Users/mcarthur/Desktop/D_Science_platzi/Pandas/Salary.csv")


#Paso 2: Analisis Explortorio de Datos (EDA)
print(Salary.head())
Salary.info()
print(Salary.describe())
print(Salary.shape)

#Análisis de valores nulos
print("\nValores nulos por columna:")
print(Salary.isnull().sum())

print("\nTotal de valores nulos en el dataset:")
print(Salary.isnull().sum().sum())

#Analizando Salario promedio, salario minimo y maximo y rango de experiencia

mean_salary = Salary["Salary"].mean()
print("La media es: ", mean_salary)

max_salary = Salary["Salary"].max()
print("El salario maximo es: ", max_salary)

min_salary = Salary["Salary"].min()
print("El salario minimo es: ", min_salary)

range_experience = Salary["YearsExperience"].max()-Salary["YearsExperience"].min()
print("Rango de experiencia:", range_experience)

#Matriz de correlación
correlation_matrix = Salary.corr()
print("\nMatriz de correlación:")
print(correlation_matrix)

#Filtrar correlaciones superiores a 0.80
high_corr = correlation_matrix[abs(correlation_matrix) > 0.80]
print("\nCorrelaciones superiores a 0.80:")
print(high_corr)

#Visualizar relación experiencia-salario
plt.scatter(Salary["YearsExperience"], Salary["Salary"])
plt.xlabel("Years Of Experience")
plt.ylabel("Salary")
plt.title("Salary vs Years Of Experience")
plt.show()

#Paso 3: Preparar los datos

X = Salary[["YearsExperience"]]
y = Salary["Salary"]

#Dividir en entrenamiento y prueba

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Paso 4: Entrenar el modelo base (Regresión lineal)
model = LinearRegression()
model.fit(X_train, y_train)

print("Coeficiente (pendiente): ", model.coef_[0])
print("Intercepto: ", model.intercept_)

#Paso 5: Predicción y Evaluación

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("MAE:", mean_absolute_error(y_test, y_pred)) #error promedio del modelo
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)
print("R2:", r2_score(y_test, y_pred)) #Que porcentaje de la variabilidad explica el modelo

#Paso 6: Visualizar el modelo

plt.scatter(X, y, label="Datos reales")
X_sorted = X.sort_values(by="YearsExperience")
plt.plot(X_sorted, model.predict(X_sorted), label="Regresión Lineal")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()

#Paso 7: Validación cruzada

scores = cross_val_score(model, X, y, cv=5, scoring="r2")
scores
scores.mean()

print("Scores CV:", scores)
print("R2 promedio CV:", scores.mean())

#Paso 8: Comparar un Ridge y Lasso

ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)

ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)

ridge.score(X_test, y_test)
lasso.score(X_test, y_test)

print("Ridge R2:", ridge.score(X_test, y_test))
print("Lasso R2:", lasso.score(X_test, y_test))

#Paso 9: Simulador de salario
 
def predict_salary(years_experience):
    years_experience = pd.DataFrame({"YearsExperience": [years_experience]})
    salary = model.predict(years_experience)
    return salary[0]