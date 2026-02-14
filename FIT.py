import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
# Cargar datos
df = pd.read_csv("fitness_dataset.csv")

print(df.corr(numeric_only=True)["is_fit"].sort_values(ascending=False))

df["sleep_hours"] = df["sleep_hours"].fillna(df["sleep_hours"].mean())

# Verificar valores nulos
print("Valores nulos por columna:")
print(df.isnull().sum())

# Verificar balance de la variable objetivo
print("\nConteo de la variable is_fit:")
print(df["is_fit"].value_counts())


# Convertir categóricas
df["smokes"] = df["smokes"].map({"yes":1, "no":0})
df["gender"] = df["gender"].map({"male":1, "female":0})

# Separar X y y
X = df.drop("is_fit", axis=1)
y = df["is_fit"]


from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y),
    y=y
)

class_weights = dict(enumerate(class_weights))
print("Pesos de clase:", class_weights)


# División
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Escalar
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[early_stop]
)

# Evaluación
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy final:", accuracy)


# Obtener probabilidades
y_pred_prob = model.predict(X_test)

# Convertir probabilidades a 0 o 1
y_pred = (y_pred_prob > 0.5).astype(int)

"""
#Evaluación con Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

rf_accuracy = rf.score(X_test, y_test)
print("Random Forest Accuracy:", rf_accuracy)
"""

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:")
print(cm)

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

#Generar matriz de confusión con seaborn
plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.title("Matriz de Confusión - Red Neuronal")
plt.show()
