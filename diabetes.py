import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

data = pd.read_csv("/Users/mcarthur/Desktop/diabetes.csv")
print(data.head())
#data.info()
#data.isnull().sum()
print(data.shape)

#Creación de histograma

#Ajustar el tamaño de la fuente
plt.rcParams["font.size"] = 15
#Crear una figura y ajustar su tamaño
f = plt.figure(figsize=(8,4))
#Crear un subplot o subrama - al ser una sola figura es 1,1,1
ax = f.add_subplot(1,1,1)
#Grafica tus datos usando "hist". Pasa el objeto "ax" a pandas. Agrega un borde negro con un groser de 2
data["Outcome"].hist(ax=ax, edgecolor = "black", linewidth=2)
#Establece los limites en el eje X
ax.set_xlim([-0.5, 1.5])
#Establece la frecuencia de tick. Tenemos 0 y 1 que corresponden a Si y No respectivamente.
ax.set_xticks([0, 1])
#Etiquetar xtick labels
ax.set_xticklabels(["N", "Y"])
#Crea el Titulo
ax.set_title("Diabetes Y/N?")
#Establece la etiqueta del eje X
ax.set_xlabel("Answer")
#Establece la etiqueta del eje Y
ax.set_ylabel("Count")
#Establece los limites superior/inferior del eje Y
ax.set_ylim([0, 510])
#Hace que las cosas sean bonitas, no es necesario, pero se ajuta al tamaño de la figura.
f.tight_layout()

f = plt.figure(figsize=(8,4))
ax = f.add_subplot(1,1,1)
data["Age"].hist(ax=ax, edgecolor="black", linewidth=2)
ax.set_title("Age range of patients")
ax.set_ylim([0,510])
ax.set_xlabel("Age")
ax.set_ylabel("Count")
f.tight_layout()

#División de Datos
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

#Split entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)

#Escalar los datos (muy importante en regresión logistica)
scaler = StandardScaler() #Normaliza las variables para que ninguna domine por tener valores grandes
X_train = scaler.fit_transform(X_train) #fit solo en train (evita leakage) que el test influya en el escalado
X_test = scaler.transform(X_test) #transform en test

#Entrenar el modelo
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

#Predicciones
y_pred = model.predict(X_test)

#Métricas
print("="*45)
print(f" Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("="*45)
#Muestra precisión, recall y F1 por clase
print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))

#Matriz de confusión
#Visualiza cuantos errores y aciertos tuvo el modelo
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Diabetes","Diabetes"],
            yticklabels=["No Diabetes", "Diabetes"])
plt.title("Matriz de Confusión")
plt.ylabel("Real")
plt.xlabel("Predicho")
plt.tight_layout()
plt.show()