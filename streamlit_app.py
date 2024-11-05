import warnings
warnings.filterwarnings('ignore')

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, hamming_loss, f1_score
import catboost
from catboost import CatBoostClassifier
import numpy as np

# Carregar modelos salvos
preprocessing_components = joblib.load('preprocessing_components.joblib')
le_type = preprocessing_components['label_encoder_type']
scaler = preprocessing_components['scaler']
ohe = preprocessing_components['one_hot_encoder']

# Carregar os classificadores individuais
model_paths = [
    'modelo_catboost_falhas_Failure_Type_HDF.joblib',
    'modelo_catboost_falhas_Failure_Type_OSF.joblib',
    'modelo_catboost_falhas_Failure_Type_PWF.joblib',
    'modelo_catboost_falhas_Failure_Type_RNF.joblib',
    'modelo_catboost_falhas_Failure_Type_TWF.joblib'
]
final_classifiers = {path.split('_')[-1].split('.')[0]: joblib.load(path) for path in model_paths}

# Função para fazer predição
def predict_failure(input_data):
    try:
        input_data_scaled = scaler.transform(input_data)
        predictions = {}
        for label, clf in final_classifiers.items():
            prediction = clf.predict(input_data_scaled)
            predictions[label] = prediction[0]
        falhas = [label for label, pred in predictions.items() if pred == 1]
        return falhas if falhas else ["Sem falhas detectadas"]
    except Exception as e:
        return [f"Erro ao fazer a previsão: {e}"]

# Código do aplicativo Streamlit
import streamlit as st
import base64
import matplotlib.pyplot as plt
import seaborn as sns

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

# Agrupando as entradas como array
input_data = np.array([[type_encoded, air_temp, process_temp, rot_speed, torque, tool_wear]])

# Botão de previsão
if st.button("🔍 Prever Falha"):
    resultado = predict_failure(input_data)
    resultado_str = ', '.join(resultado)
    st.markdown(
        f"""
        <div style="padding:10px; border-radius:5px; background-color: {'#cb0000' if 'Sem falhas detectadas' not in resultado else '#26b500'};">
            <h3 style="text-align: center; color: white;">Resultado da Previsão</h3>
            <p style="text-align: center; font-size: 20px; font-weight: bold;">{resultado_str}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

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