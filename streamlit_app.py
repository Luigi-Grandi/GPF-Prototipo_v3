import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib  # Para carregar o scaler salvo

# Carregar o modelo LSTM previamente salvo
model = load_model('my_model2.keras')

# Carregar o scaler salvo
scaler = joblib.load('scaler.pkl')  # Supondo que você tenha salvo o scaler ao treinar o modelo

# Título do aplicativo
st.title("Previsão de Falha de Máquina com LSTM")

# Entradas para o usuário
st.header("Insira os dados da máquina:")

# Campos de entrada atualizados
type_value = st.selectbox("Tipo da Máquina (Type)", ["L", "M", "H"])  # Adapte para os valores possíveis em Type
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

# Verificar se o scaler foi carregado corretamente
if scaler is not None:
    input_data_scaled = scaler.transform(input_data)
else:
    st.error("Erro ao carregar o scaler. Verifique se 'scaler.pkl' está disponível.")

# Preparando a entrada no formato de sequência esperado pelo LSTM (1, 10, número de features)
# Aqui assumo que você precisará de 10 timesteps; podemos replicar a entrada para criar uma sequência realista
time_steps = 10
X_input = np.tile(input_data_scaled, (time_steps, 1))  # Replicando os dados da entrada para formar uma sequência
X_input = X_input.reshape((1, time_steps, input_data_scaled.shape[1]))

# Realizando a previsão quando o botão é pressionado
if st.button("Prever Falha"):
    try:
        # Fazendo a previsão
        prediction = model.predict(X_input)
        predicted_class = int(np.round(prediction[0][0]))  # Supondo uma saída binária (0 ou 1)
        resultado = "Falha" if prediction >= 0.1 else "Sem Falha"
        
        # Mostrando os resultados da previsão
        #st.write(f"Resultado númerico da Previsão: {prediction[0][0]}")
        st.write(f"Resultado da Previsão: {resultado}")
    except Exception as e:
        st.error(f"Erro ao fazer a previsão: {e}")