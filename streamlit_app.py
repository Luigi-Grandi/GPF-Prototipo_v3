import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o modelo LSTM previamente salvo
model = load_model('my_model2.keras')

# Carregar o scaler salvo
scaler = joblib.load('scaler.pkl')

# Carregar o arquivo CSV
data = pd.read_csv('data/galds.csv')

# Título do aplicativo
st.title("Preevisão de Falha de Máquina com LSTM")

# Seção de gráficos exploratórios
st.header("Análise Exploratória dos Dados")

# Gráfico 1: Distribuição de temperatura do ar em função do tipo de máquina
st.subheader("Distribuição de Temperatura do Ar por Tipo de Máquina")
fig, ax = plt.subplots()
sns.boxplot(data=data, x='Type', y='Air temperature [K]', ax=ax)
st.pyplot(fig)

# Gráfico 2: Rotational speed vs Torque colorido por Machine failure
st.subheader("Velocidade Rotacional vs Torque (Colorido por Falha)")
fig, ax = plt.subplots()
sns.scatterplot(data=data, x='Rotational speed [rpm]', y='Torque [Nm]', hue='Machine failure', palette='coolwarm', ax=ax)
st.pyplot(fig)

# Gráfico 3: Correlação entre variáveis
st.subheader("Matriz de Correlação")
fig, ax = plt.subplots(figsize=(10, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Entradas para o usuário
st.header("Insira os dados da máquina:")

# Campos de entrada atualizados
type_value = st.selectbox("Tipo da Máquina (Type)", ["L", "M", "H"])
air_temp = st.number_input("Temperatura do Ar [K]", min_value=0.0, format="%.2f")
process_temp = st.number_input("Temperatura do Processo [K]", min_value=0.0, format="%.2f")
rot_speed = st.number_input("Velocidade Rotacional [rpm]", min_value=0, format="%d")
torque = st.number_input("Torque [Nm]", min_value=0.0, format="%.2f")
tool_wear = st.number_input("Desgaste da Ferramenta [min]", min_value=0, format="%d")

# Mapeamento e conversão dos dados de entrada para valores numéricos
type_mapping = {"L": 0, "M": 1, "H": 2}
type_encoded = type_mapping[type_value]

# Agrupando as entradas como array e padronizando usando o scaler treinado
input_data = np.array([[type_encoded, air_temp, process_temp, rot_speed, torque, tool_wear]])

if scaler is not None:
    input_data_scaled = scaler.transform(input_data)
else:
    st.error("Erro ao carregar o scaler. Verifique se 'scaler.pkl' está disponível.")

# Preparando a entrada no formato de sequência esperado pelo LSTM (1, 10, número de features)
time_steps = 10
X_input = np.tile(input_data_scaled, (time_steps, 1))
X_input = X_input.reshape((1, time_steps, input_data_scaled.shape[1]))

# Realizando a previsão quando o botão é pressionado
if st.button("Prever Falha"):
    try:
        prediction = model.predict(X_input)
        predicted_class = int(np.round(prediction[0][0]))
        resultado = "Falha" if prediction >= 0.1 else "Sem Falha"
        st.write(f"Resultado da Previsão: {resultado}")
    except Exception as e:
        st.error(f"Erro ao fazer a previsão: {e}")
