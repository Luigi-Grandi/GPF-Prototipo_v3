import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import time

# Configurações para o Streamlit e carregamento da imagem de logotipo
logo_path = "data/logo.jpg"  # Caminho para a imagem do logotipo
logo_ext = "jpg"  # Extensão do logotipo
logo_base64 = base64.b64encode(open(logo_path, "rb").read()).decode()

st.markdown(
    """
    <style>
    .header-container { ... }
    </style>
    """,
    unsafe_allow_html=True
)

# Cabeçalho com logotipo e título
st.markdown(
    f"""
    <div class="header-container">
        <img src="data:image/{logo_ext};base64,{logo_base64}" alt="Logo">
        <h1>Gestor Preditivo de Falhas</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Carregar o modelo e scaler
model = load_model('my_model2.keras')
scaler = joblib.load('scaler.pkl')
data = pd.read_csv('data/galds.csv')

# Entradas do usuário
st.title("🔧 Dashboard da previsão de Falha de Máquina")
st.write("Bem-vindo ao sistema de previsão de falhas! Insira os dados da máquina e explore as análises gráficas.")

# Entradas de parâmetros
st.sidebar.title("Configurações e Entrada de Dados")
type_value = st.sidebar.selectbox("Tipo da Máquina (Type)", ["L", "M", "H"])
air_temp = st.sidebar.number_input("Temperatura do Ar [K]", min_value=0.0, format="%.2f")
process_temp = st.sidebar.number_input("Temperatura do Processo [K]", min_value=0.0, format="%.2f")
rot_speed = st.sidebar.number_input("Velocidade Rotacional [rpm]", min_value=0, format="%d")
torque = st.sidebar.number_input("Torque [Nm]", min_value=0.0, format="%.2f")
tool_wear = st.sidebar.number_input("Desgaste da Ferramenta [min]", min_value=0, format="%d")

# Mapeamento e conversão dos dados de entrada
type_mapping = {"L": 0, "M": 1, "H": 2}
type_encoded = type_mapping[type_value]

# Agrupando as entradas e padronizando
input_data = np.array([[type_encoded, air_temp, process_temp, rot_speed, torque, tool_wear]])
input_data_scaled = scaler.transform(input_data) if scaler is not None else st.error("Erro ao carregar o scaler.")

# Preparando a entrada para o LSTM (1, 10, número de features)
time_steps = 10
X_input = np.tile(input_data_scaled, (time_steps, 1)).reshape((1, time_steps, input_data_scaled.shape[1]))

# Botão de previsão
if st.button("🔍 Prever Falha"):
    prediction = model.predict(X_input)
    resultado = "Falha" if prediction >= 0.05 else "Sem Falha"
    st.markdown(
        f"""
        <div style="padding:10px; border-radius:5px; background-color: {'#cb0000' if resultado == 'Falha' else '#26b500'};">
            <h3 style="text-align: center; color: white;">Resultado da Previsão</h3>
            <p style="text-align: center; font-size: 20px; font-weight: bold;">{resultado}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Análise de correlação e visualização de dados
with st.expander("Veja mais análises de correlação"):
    st.header("📊 Análise Geral dos Dados")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Distribuição de Temperatura do Ar por Tipo de Máquina")
        fig1, ax1 = plt.subplots()
        sns.boxplot(data=data, x='Type', y='Air temperature [K]', ax=ax1)
        st.pyplot(fig1)
    with col2:
        st.subheader("📉 Velocidade Rotacional vs Torque (Colorido por Falha)")
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=data, x='Rotational speed [rpm]', y='Torque [Nm]', hue='Machine failure', palette='coolwarm', ax=ax2)
        st.pyplot(fig2)

# Previsões contínuas para cada instância no arquivo de dados
with st.expander("Análise Contínua de Máquina"):
    result_div = st.empty()
    chart_placeholder = st.empty()
    predictions = []
    air_temp_values, process_temp_values, rot_speed_values, torque_values, tool_wear_values = [], [], [], [], []

    for index, row in data.iterrows():
        type_encoded = type_mapping[row['Type']]
        input_data = np.array([[type_encoded, row['Air temperature [K]'], row['Process temperature [K]'], row['Rotational speed [rpm]'], row['Torque [Nm]'], row['Tool wear [min]']]])
        input_data_scaled = scaler.transform(input_data)
        X_input = np.tile(input_data_scaled, (10, 1)).reshape((1, 10, input_data_scaled.shape[1]))

        # Fazer a previsão
        prediction_value = model.predict(X_input)[0][0]
        predictions.append(prediction_value)

        # Gráficos para cada parâmetro
        air_temp_values.append(row['Air temperature [K]'])
        process_temp_values.append(row['Process temperature [K]'])
        rot_speed_values.append(row['Rotational speed [rpm]'])
        torque_values.append(row['Torque [Nm]'])
        tool_wear_values.append(row['Tool wear [min]'])

        # Limitar o número de pontos mostrados nos gráficos
        if len(predictions) > 10:
            predictions.pop(0)
            air_temp_values.pop(0)
            process_temp_values.pop(0)
            rot_speed_values.pop(0)
            torque_values.pop(0)
            tool_wear_values.pop(0)

        # Gráfico de previsões
        fig_predictions, ax_predictions = plt.subplots()
        ax_predictions.plot(predictions, marker='o', color='blue')
        ax_predictions.set_title("Previsões de Falhas ao Longo do Tempo")
        ax_predictions.set_xlabel("Instância")
        ax_predictions.set_ylabel("Probabilidade de Falha")
        chart_placeholder.pyplot(fig_predictions)

        # Gráficos de parâmetros
        col1, col2 = st.columns(2)
        with col1:
            fig_air_temp, ax_air_temp = plt.subplots()
            ax_air_temp.plot(air_temp_values, marker='o', color='orange')
            ax_air_temp.set_title("Temperatura do Ar [K]")
            st.pyplot(fig_air_temp)
        time.sleep(3)

# Esse é o código completo ajustado até onde possível dentro dos limites de exibição. Verifique se tudo está em seu lugar, especialmente o caminho de arquivos como o logotipo e o scaler, e considere otimizar o código para maior legibilidade.
