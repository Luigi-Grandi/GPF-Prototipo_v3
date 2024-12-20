
# Importar Bibliotecas
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path

# Remover Avisos

import warnings
warnings.filterwarnings('ignore')

# Importando o dataset
path = Path(__file__).parent/'data/galds.csv'

with open(path, "rt") as f:
  df = pd.read_csv(path)

df.head()

# Mostrar Números de Valores nulos por séries

df.isnull().sum()
print(df.dtypes)
# Verifiquei e a coluna UDI e Product ID é inútil
# Removendo as colunas 'UDI' e 'Product ID'
df = df.drop(columns=['UDI', 'Product ID'])

df.head ()
# Transformar em número as colunas Strings

from sklearn.preprocessing import LabelEncoder

# Aplicando LabelEncoder na coluna 'Type'
le_type = LabelEncoder()
df['Type_encoded'] = le_type.fit_transform(df['Type'])



# Removendo as colunas originais
df = df.drop(columns=['Type'])

# Exibindo o DataFrame atualizado
print(df)


# Padronizando as features contínuas
scaler = StandardScaler()
continuous_columns = [
    'Air temperature [K]', 'Process temperature [K]',
    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'
]
df[continuous_columns] = scaler.fit_transform(df[continuous_columns])

# Transformando a coluna 'Machine failure' em valores binários
df['Machine failure'] = df['Machine failure'].astype(int)

# Estruturando os dados para o LSTM
# Supondo que você queira usar uma janela de 10 timesteps
def create_sequences(data, target, time_steps=10):
    sequences = []
    targets = []

    for i in range(len(data) - time_steps):
        sequences.append(data[i:i + time_steps])
        targets.append(target[i + time_steps])

    return np.array(sequences), np.array(targets)

# Definindo as features e o alvo
X = df.drop(columns=['Machine failure']).values
y = df['Machine failure'].values

# Criando sequências
X_seq, y_seq = create_sequences(X, y, time_steps=10)

# Dividindo os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Definindo o modelo LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))  # Usando sigmoid para classificação binária

# Compilando o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinando o modelo
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
# Avaliando o modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')
model.save('modelo_lstm.h5')
# Carregue o modelo previamente treinado
# Exemplo: model = tf.keras.models.load_model('meu_modelo.h5')