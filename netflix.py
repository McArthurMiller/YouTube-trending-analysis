import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

#Pandas maneja datos
#train_test_split -> dividir datos en entrenamiento y prueba
#OneHotEncodere -> convierte texto en números
#LogisticRegression -> modelo
#metrics -> evalua resultados

#Cargando el Dataset
#Leemos el archivo CSV y lo convertimos en un DF 
df = pd.read_csv("/Users/mcarthur/Desktop/Tarea Onmex/Modelo2/netflix_titles.csv")
print(df.head())

print(df.info())
print(df.isnull().sum())

#limpieza basica
#En este paso separamos los numeros de las letras de la columna "duration"
df["duration_num"] = df["duration"].str.extract("(\d+)")
df["duration_num"] = pd.to_numeric(df["duration_num"], errors="coerce")

#Eliminar nulos
#Los modelos de sklearn no aceptan valores nulos, se dejan solo filas completas
df_model = df[["type","release_year","rating","duration_num"]]
df_model = df_model.dropna()

#Definir X y Y
#X -> variables predictoras (caracteriticas)
#Y -> variable objeto (lo que queremos predecir)(movie o tv show)
X = df_model[["release_year","rating","duration_num"]]
y = df_model["type"]

#Convertir variables categoricas
#La columna "rating" es texto, necesitamos convertirla en números usando One-Hot Encoding
#Que es One_Hot Encoding: Si rating tiene PG R TV-MA, se convierte en: rating_R, rating_TV-MA
#Valores 0 o 1, esto evita que el modelo piense que "R > PG"
X = pd.get_dummies(X, columns=["rating"], drop_first=True)

#dividir los datos
#Esto quiere decir que 80% datos -> entrenar modelo
#20% datos -> probar modelo
#random_state -> reproducibilidad
#NUNCA SE ENTRENA Y EVALUA CON LOS MISMOS DATOS
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Entrenar modelo
#El modelo aprende relaciones matematicas entre release_year + duration + rating
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#Hacer predicciones
#Ahora el modelo intenta adivinar el tipo usando datos que nunca vio
y_pred = model.predict(X_test)

#Evaluar modelo
#Metricas importantes:
#Accuracy -> porcentaje de aciertos
#Precision -> que tan preciso es cuando predice algo
#Recall -> que tanto detecta correctamente
#F1-score -> balance entre precision y recall
print("Accuracy: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Conceptos importantes:
# Clasificación supervisada
#Variable objetivo
#Feature engineering
#One-Hot Encoding
#Train/Test Split
#Logistic Regression
#Evaluación

#Cuando un modelo da casi 100% en un problema relativamente simple, pueden pasar 3 cosas:
#1 El problema es realmente facil
#2 Hay desbalance de clases
#3 Hay fuga de información(data leakage)

#Revisión
#Revisión de balance de clases

print(df_model["type"].value_counts())

#IMPORTANTE:
#Cuando un feature contiene información casi directa de la variable objetivo,
#el modelo parece "demasiado bueno", esto en producción puede ser peligroso

print(classification_report(y_test, y_pred))

#ENTRENAR RANDOM FOREST

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

#PREDICCION

y_pred_rf = rf.predict(X_test)

#Evaluación

print("Accuracy: ", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

#Lo importante: Feature Importance

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": rf.feature_importances_
}).sort_values(by="importance", ascending=False)

print(importance)