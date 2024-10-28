import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Carregar o modelo LSTM previamente salvo
model = load_model('modelo_lstm.h5')

# Título do aplicativo
st.title("Previsão de Falha de Máquina com LSTM")

# Entradas para o usuário
st.header("Insira os dados da máquina:")

# Cada campo de entrada
type_value = st.selectbox("Tipo da Máquina (Type)", ["A", "B", "C"])  # Adapte para os valores possíveis em `Type`
air_temp = st.number_input("Temperatura do Ar [K]", min_value=0.0, format="%.2f")
process_temp = st.number_input("Temperatura do Processo [K]", min_value=0.0, format="%.2f")
rot_speed = st.number_input("Velocidade Rotacional [rpm]", min_value=0, format="%d")
torque = st.number_input("Torque [Nm]", min_value=0.0, format="%.2f")
tool_wear = st.number_input("Desgaste da Ferramenta [min]", min_value=0, format="%d")
twf = st.selectbox("Falha de Ferramenta (TWF)", [0, 1])
hdf = st.selectbox("Falha de Head (HDF)", [0, 1])
pwf = st.selectbox("Falha de Potência (PWF)", [0, 1])
osf = st.selectbox("Falha no Sistema de Sobrecarga (OSF)", [0, 1])
rnf = st.selectbox("Falha de Recipiente (RNF)", [0, 1])

# Mapeamento e conversão dos dados de entrada para valores numéricos
type_mapping = {"A": 0, "B": 1, "C": 2}  # Use o mapeamento correto para seu modelo
type_encoded = type_mapping[type_value]

# Agrupando as entradas como array e padronizando
scaler = StandardScaler()
input_data = np.array([[type_encoded, air_temp, process_temp, rot_speed, torque, tool_wear, twf, hdf, pwf, osf, rnf]])
input_data_scaled = scaler.fit_transform(input_data)  # Supondo que o modelo foi treinado com dados padronizados

# Preparando a entrada no formato de sequência esperado pelo LSTM (1, 10, número de features)
time_steps = 10
X_input = np.array([input_data_scaled for _ in range(time_steps)])  # Usando valores repetidos para simular uma sequência
X_input = np.expand_dims(X_input, axis=0)

# Removendo o eixo adicional para ajustar à forma correta
X_input = np.squeeze(X_input, axis=2)  # Agora a forma será (1, 10, 11)

# Realizando a previsão quando o botão é pressionado
if st.button("Prever Falha"):
    prediction = model.predict(X_input)
    predicted_class = int(np.round(prediction[0][0]))  # Supondo uma saída binária
    resultado = "Falha" if predicted_class == 1 else "Sem Falha"
    st.write(f"Resultado da Previsão: **{resultado}**")

