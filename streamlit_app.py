import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import base64

# Carregar a imagem do logotipo
logo_path = "data/logo.jpg"  # Caminho para a imagem do logotipo
logo_ext = "jpg"  # Extensão do logotipo

# Codificar a imagem do logotipo em Base64
logo_base64 = base64.b64encode(open(logo_path, "rb").read()).decode()

# Configurações de estilo personalizado com CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #d8bfd8; /* Tom de roxo claro */
    }
    .header-container {
        display: flex;
        align-items: left;
        justify-content: left;
        padding: 10px;
        //background-color: #4b0082; /* Fundo do cabeçalho em roxo escuro */
        color: white;
        border-radius: 8px;
    }
    .header-container img {
        width: 180px; /* Ajuste o tamanho do logotipo */
        height: auto;
        margin-right: 10px;
    }
    .header-container h1 {
        color: #4b0082;
        font-size: 60px;
    }
    h1, h2, h3 {
        color: #1f77b4; /* Nova cor dos títulos */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Cabeçalho com logotipo e título
st.markdown(
    f"""
    <div class="header-container">
        <img src="data:image/{logo_ext};base64,{logo_base64}" alt="Logo">
        <h1>Preditor de Falha GPF</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Adicione um espaço para evitar que o conteúdo inicial fique atrás do cabeçalho fixo
st.markdown("<div style='margin-top: 80px;'></div>", unsafe_allow_html=True)

# Carregar o modelo LSTM previamente salvo
model = load_model('my_model2.keras')

# Carregar o scaler salvo
scaler = joblib.load('scaler.pkl')  # Supondo que você tenha salvo o scaler ao treinar o modelo

# Carregar o arquivo CSV
data = pd.read_csv('data/galds.csv')

# Título e introdução do aplicativo
st.title("🔧 Previsão de Falha de Máquina com LSTM")
st.write("Bem-vindo ao sistema de previsão de falhas! Insira os dados da máquina e explore as análises gráficas.")

# Menu lateral para as entradas do usuário
st.sidebar.title("Configurações e Entrada de Dados")
type_value = st.sidebar.selectbox("Tipo da Máquina (Type)", ["L", "M", "H"])
air_temp = st.sidebar.number_input("Temperatura do Ar [K]", min_value=0.0, format="%.2f")
process_temp = st.sidebar.number_input("Temperatura do Processo [K]", min_value=0.0, format="%.2f")
rot_speed = st.sidebar.number_input("Velocidade Rotacional [rpm]", min_value=0, format="%d")
torque = st.sidebar.number_input("Torque [Nm]", min_value=0.0, format="%.2f")
tool_wear = st.sidebar.number_input("Desgaste da Ferramenta [min]", min_value=0, format="%d")

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
time_steps = 10
X_input = np.tile(input_data_scaled, (time_steps, 1))
X_input = X_input.reshape((1, time_steps, input_data_scaled.shape[1]))

# Botão de previsão
if st.button("🔍 Prever Falha"):
    try:
        # Fazendo a previsão
        prediction = model.predict(X_input)
        resultado = "Falha" if prediction >= 0.1 else "Sem Falha"
        
        # Exibindo o resultado em um cartão de destaque
        st.markdown(
            f"""
            <div style="padding:10px; border-radius:5px; background-color: {'#FFCCCC' if resultado == 'Falha' else '#CCFFCC'};">
                <h3 style="text-align: center;">Resultado da Previsão</h3>
                <p style="text-align: center; font-size: 20px; font-weight: bold;">{resultado}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Erro ao fazer a previsão: {e}")

# Análise Exploratória dos Dados
st.header("📊 Análise Exploratória dos Dados")

# Dividir gráficos em colunas para melhor organização
col1, col2 = st.columns(2)

# Gráfico 1: Distribuição de temperatura do ar em função do tipo de máquina
with col1:
    st.subheader("📈 Distribuição de Temperatura do Ar por Tipo de Máquina")
    fig1, ax1 = plt.subplots()
    sns.boxplot(data=data, x='Type', y='Air temperature [K]', ax=ax1)
    st.pyplot(fig1)

# Gráfico 2: Rotational speed vs Torque colorido por Machine failure
with col2:
    st.subheader("📉 Velocidade Rotacional vs Torque (Colorido por Falha)")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=data, x='Rotational speed [rpm]', y='Torque [Nm]', hue='Machine failure', palette='coolwarm', ax=ax2)
    st.pyplot(fig2)

# Expansor para visualização da matriz de correlação
with st.expander("Veja mais análises de correlação"):
    st.subheader("🔍 Matriz de Correlação")
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    correlation_matrix = data.drop(columns=['UDI', 'Product ID', 'Type']).corr()  # Removendo colunas não numéricas
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)
