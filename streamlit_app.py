import streamlit as st
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import base64

# Definição da classe personalizada WeightedClassifierChain
class WeightedClassifierChain(ClassifierChain):
    def _init_(self, base_estimator, order='random', random_state=None, weights=None):
        super()._init_(base_estimator=base_estimator, order=order, random_state=random_state)
        self.weights = weights

    def fit(self, X, Y, **fit_params):
        if self.weights is None:
            raise ValueError("Weights must be provided for each class.")
        if len(self.weights) != Y.shape[1]:
            raise ValueError("Number of weights must match number of classes.")

        self.estimators_ = [clone(self.base_estimator) for _ in range(Y.shape[1])]
        for i, estimator in enumerate(self.estimators_):
            estimator.set_params(scale_pos_weight=self.weights[i])

        return super().fit(X, Y, **fit_params)

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

# Carregar o pipeline salvo (inclui scaler e modelo)
@st.cache_resource
def load_pipeline():
    pipeline = joblib.load('model_pipeline.joblib')
    return pipeline

pipeline = load_pipeline()

# Carregar o arquivo CSV para análises adicionais (opcional)
data = pd.read_csv('data/galds.csv')

# Título e introdução do aplicativo
st.title("🔧 Dashboard da Previsão de Falhas de Máquina")
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

# Criação das novas features conforme engenharia de features do modelo
temp_diff = process_temp - air_temp
power = torque * rot_speed
wear_torque = tool_wear * torque

# Agrupando as entradas como array
input_data = np.array([[type_encoded, air_temp, process_temp, rot_speed, torque, tool_wear, temp_diff, power, wear_torque]])

# Botão de previsão
if st.button("🔍 Prever Falhas"):
    try:
        # Fazendo a previsão multilabel
        y_pred = pipeline.predict(input_data)
        # y_pred é um array binário indicando a presença de cada falha

        # Obter os nomes das classes
        classes = pipeline.named_steps['classifier'].classes_

        # Mapear as predições para os nomes das classes
        predicted_failures = [cls for cls, pred in zip(classes, y_pred[0]) if pred == 1]

        # Exibindo o resultado
        if predicted_failures:
            falhas = ', '.join(predicted_failures)
            st.markdown(
                f"""
                <div style="padding:10px; border-radius:5px; background-color: #cb0000;">
                    <h3 style="text-align: center; color: white;">Falhas Previstas</h3>
                    <p style="text-align: center; font-size: 20px; font-weight: bold;">{falhas}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="padding:10px; border-radius:5px; background-color: #26b500;">
                    <h3 style="text-align: center; color: white;">Sem Falhas Previstas</h3>
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

# Análises adicionais (opcional)
with st.expander("Análises Adicionais"):
    st.header("🔍 Análises Complementares")

    # Exemplo: Distribuição das classes de falha
    st.subheader("📊 Distribuição das Classes de Falha")
    # Como estamos fazendo predições em tempo real, podemos calcular a distribuição com base nas predições anteriores ou nos dados históricos
    # Aqui, usarei os dados históricos
    failure_columns = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    failure_counts = data[failure_columns].sum()
    st.bar_chart(failure_counts)

    # Exemplo: Importância das Features
    st.subheader("📈 Importância das Features")
    # Obter a média das importâncias das features de todos os classificadores
    feature_importances = np.mean([
        estimator.feature_importances_ for estimator in pipeline.named_steps['classifier'].estimators_
    ], axis=0)
    feature_names = ['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
                     'Torque [Nm]', 'Tool wear [min]', 'Temp_Diff', 'Power', 'Wear_Torque']
    feature_importances_series = pd.Series(feature_importances, index=feature_names).sort_values(ascending=False)
    fig3, ax3 = plt.subplots(figsize=(10,6))
    sns.barplot(x=feature_importances_series, y=feature_importances_series.index, ax=ax3)
    ax3.set_xlabel("Importância das Features")
    ax3.set_ylabel("Features")
    st.pyplot(fig3)