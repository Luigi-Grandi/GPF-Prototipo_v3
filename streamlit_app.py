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

# Carregar a imagem do logotipo
logo_path = "data/logo.jpg"  # Caminho para a imagem do logotipo
logo_ext = "jpg"  # Extensão do logotipo

# Codificar a imagem do logotipo em Base64
logo_base64 = base64.b64encode(open(logo_path, "rb").read()).decode()

# Configurações de estilo personalizado com CSS
st.markdown(

           """
    <style>
    .header-container {
        display: flex;
        align-items: center;  /* Centraliza verticalmente */
        justify-content: flex-start;  /* Alinhamento à esquerda */
        padding: 10px;
        color: white;
        border-radius: 8px;
        max-width: 100%; /* Limita a largura do cabeçalho */
        overflow: hidden; /* Oculta o conteúdo que excede */
        flex-wrap: wrap;  /* Permite que os itens se movam para a linha seguinte se necessário */
    }
    .header-container img {
        width: auto;
        max-width: 120px; /* Ajuste o tamanho do logotipo */
        height: auto; /* Mantém a proporção da imagem */
        margin-right: 10px;
        border-radius: 10px;
    }
    .header-container h1 {
        color: #FFFFFF;
        font-size: 40px; /* Tamanho padrão */
        white-space: nowrap; /* Impede que o texto quebre em várias linhas */
    }
    @media (max-width: 600px) { /* Ajustes para telas menores */
        .header-container h1 {
            font-size: 15px; /* Reduz o tamanho da fonte em telas pequenas */
        }
        .header-container img {
            width: auto;
            max-width: 40px; /* Ajuste o tamanho do logotipo */
            height: auto; /* Mantém a proporção da imagem */
            margin-right: 10px;
            border-radius: 10px;
        }
        
    }
    h1, h2, h3 {
        color: #1f77b4; /* Nova cor dos títulos */
    }
    div.stButton button 
    {
        flex: 1;
        flex-align: center;
        align-self: center;
        width: auto;
        cursor: pointer;
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
        <h1>Gestor Preditivo de Falhas</h1>
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
st.title("🔧 Dashboard da previsão de Falha de Máquina")
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
        resultado = "Falha" if prediction >= 0.05 else "Sem Falha"
        
        # Exibindo o resultado em um cartão de destaque
        st.markdown(
            f"""
            <div style="padding:10px; border-radius:5px; background-color: {'#cb0000' if resultado == 'Falha' else '#26b500'};">
                <h3 style="text-align: center; color: white;">Resultado da Previsão</h3>
                <p style="text-align: center; font-size: 20px; font-weight: bold;">{resultado}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Erro ao fazer a previsão: {e}")

# Expansor para visualização da matriz de correlação
with st.expander("Veja mais análises de correlação"):
    # Análise Exploratória dos Dados
    st.header("📊 Análise Geral dos Dados")

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

# Carregar o arquivo CSV
data = pd.read_csv('data/galds.csv')

# Mapeamento dos tipos
type_mapping = {"L": 0, "M": 1, "H": 2}

with st.expander("Analise Continua de Máquina: "):
        # Configuração do gráfico interativo
    st.subheader("📉 Monitoramento Contínuo de Previsão de Falhas")
    prediction_fig, prediction_ax = plt.subplots()
    predictions = []  # Lista para armazenar os resultados de cada previsão
    indices = []  # Lista para armazenar os índices (ou instâncias) das previsões

    # Função para fazer a previsão e atualizar o gráfico
    def fazer_previsao(row, linha_atual):
        # Preparar os dados da linha
        type_encoded = type_mapping[row['Type']]
        air_temp = row['Air temperature [K]']
        process_temp = row['Process temperature [K]']
        rot_speed = row['Rotational speed [rpm]']
        torque = row['Torque [Nm]']
        tool_wear = row['Tool wear [min]']
        input_data = np.array([[type_encoded, air_temp, process_temp, rot_speed, torque, tool_wear]])

        # Padronizar e preparar a entrada para o LSTM
        input_data_scaled = scaler.transform(input_data)
        X_input = np.tile(input_data_scaled, (10, 1))
        X_input = X_input.reshape((1, 10, input_data_scaled.shape[1]))

        # Fazer a previsão
        prediction = model.predict(X_input)[0][0]  # Captura o valor da previsão
        resultado = "Falha" if prediction >= 0.05 else "Sem Falha"
        
        # Exibir o resultado
        result_div.markdown(
            f"""
            <div style="margin: 20px; padding:10px; border-radius: 25px; background-color: {'#cb0000' if resultado == 'Falha' else '#26b500'}; position: relative;">
                <h3 style="text-align: center; color: white;">Resultado da Previsão</h3>
                <p style="text-align: center; font-size: 20px; font-weight: bold;">{resultado}</p>
                <p style="font-size: 10px; font-weight: bold; position: absolute; bottom: 10px; right: 20px; margin: 0;">
                    Instancia: {linha_atual + 1}
                </p> 
            </div>
            """,
            unsafe_allow_html=True
        )

        # Adiciona a previsão e o índice ao gráfico
        predictions.append(prediction)
        indices.append(linha_atual + 1)
        
        # Limpa e redesenha o gráfico
        prediction_ax.clear()
        prediction_ax.plot(indices, predictions, marker='o', linestyle='-', color='b')
        prediction_ax.set_title("Evolução das Previsões de Falha")
        prediction_ax.set_xlabel("Instância")
        prediction_ax.set_ylabel("Valor da Previsão")
        prediction_ax.grid()
        st.pyplot(prediction_fig)

    # Placeholder para exibir o resultado em tempo real
    result_div = st.empty()

    # Loop para prever falhas e atualizar o gráfico a cada 3 segundos
    with st.expander("Analise Continua de Máquina"):
        for index, row in data.iterrows():
            fazer_previsao(row, index)
            time.sleep(3)  # Espera de 3 segundos entre as previsões