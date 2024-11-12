import streamlit as st
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import time
from datetime import datetime
from sklearn.base import clone
from sklearn.multioutput import ClassifierChain
from xgboost import XGBClassifier

# Definição da classe personalizada WeightedClassifierChain
class WeightedClassifierChain(ClassifierChain):
    def _init_(self, base_estimator, order='random', random_state=None, weights=None):
        """
        Classe personalizada que herda de ClassifierChain e permite definir scale_pos_weight individualmente para cada classe.

        Parameters:
        - base_estimator: O estimador base a ser usado para cada classe.
        - order: A ordem das classes na cadeia ('random' ou 'sequential').
        - random_state: Controle da aleatoriedade para reproducibilidade.
        - weights: Lista de scale_pos_weight para cada classe.
        """
        super()._init_(base_estimator=base_estimator, order=order, random_state=random_state)
        self.weights = weights  # Lista de scale_pos_weight para cada classe

    def fit(self, X, Y, **fit_params):
        """
        Ajusta o modelo às features X e ao target Y.

        Parameters:
        - X: Features de entrada.
        - Y: Targets multilabel.
        - fit_params: Parâmetros adicionais para o método fit (por exemplo, sample_weight).
        """
        if self.weights is None:
            raise ValueError("Weights must be provided for each class.")
        if len(self.weights) != Y.shape[1]:
            raise ValueError("Number of weights must match number of classes.")

        # Inicializar os estimadores com scale_pos_weight definido para cada classe
        self.estimators_ = [clone(self.base_estimator) for _ in range(Y.shape[1])]
        for i, estimator in enumerate(self.estimators_):
            estimator.set_params(scale_pos_weight=self.weights[i])

        # Chamar o método fit da classe base com os parâmetros adicionais
        return super().fit(X, Y, **fit_params)

# Definição das classes de falha
failure_classes = ['Desgaste de Ferramenta', 'Falha de Dissipação de Calor', 
                   'Falha de Potência', 'Falha de Esforço Excessivo', 'Falha Geral']

# Carregar a imagem do logotipo
logo_path = "data/logo.jpg"  # Caminho para a imagem do logotipo
logo_ext = "jpg"  # Extensão do logotipo

# Codificar a imagem do logotipo em Base64
try:
    with open(logo_path, "rb") as image_file:
        logo_base64 = base64.b64encode(image_file.read()).decode()
except FileNotFoundError:
    st.error(f"Logo não encontrado em {logo_path}. Verifique o caminho do arquivo.")
    logo_base64 = ""  # Definir como string vazia para evitar erros posteriores

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
    try:
        pipeline = joblib.load('model_pipeline.joblib')
        return pipeline
    except FileNotFoundError:
        st.error("Arquivo 'model_pipeline.joblib' não encontrado. Verifique o caminho e tente novamente.")
        return None
    except Exception as e:
        st.error(f"Erro ao carregar o pipeline: {e}")
        return None

pipeline = load_pipeline()

# Verificar se o pipeline foi carregado corretamente antes de prosseguir
if pipeline is None:
    st.stop()

# Carregar o arquivo CSV para análises adicionais (opcional)
try:
    data = pd.read_csv('data/galds.csv')
except FileNotFoundError:
    st.error("Arquivo 'galds.csv' não encontrado em 'data/galds.csv'. Verifique o caminho do arquivo.")
    data = pd.DataFrame()

# Inicializar session_state para controle do processamento
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Título e introdução do aplicativo
st.title("🔧 Dashboard da Previsão de Falhas de Máquina")
st.write("Bem-vindo ao sistema de previsão de falhas! Este aplicativo processa automaticamente as linhas do arquivo CSV a cada 3 segundos, fazendo predições de falhas de máquina.")

# Menu lateral para as entradas do usuário (Predições Individuais)
st.sidebar.title("Configurações e Entrada de Dados (Individual)")
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

# Placeholder para exibição das predições individuais
individual_pred_container = st.sidebar.empty()

# Função para fazer previsão em uma linha de dados
def fazer_previsao(row, linha_atual):
    # Preparar os dados da linha
    try:
        type_val = row['Type']
        air_temp_val = row['Air temperature [K]']
        process_temp_val = row['Process temperature [K]']
        rot_speed_val = row['Rotational speed [rpm]']
        torque_val = row['Torque [Nm]']
        tool_wear_val = row['Tool wear [min]']

        # Mapear o tipo
        type_encoded_val = type_mapping.get(type_val, 0)  # default to 0 se não encontrado

        # Criar as novas features
        temp_diff_val = process_temp_val - air_temp_val
        power_val = torque_val * rot_speed_val
        wear_torque_val = tool_wear_val * torque_val

        # Agrupar as features
        input_row = np.array([[type_encoded_val, air_temp_val, process_temp_val, rot_speed_val,
                               torque_val, tool_wear_val, temp_diff_val, power_val, wear_torque_val]])

        # Fazer a previsão
        y_pred_row = pipeline.predict(input_row)

        # Mapear as predições
        predicted_failures_row = [cls for cls, pred in zip(failure_classes, y_pred_row[0]) if pred == 1]

        # Retornar as predições
        return predicted_failures_row

    except Exception as e:
        return f"Erro ao processar a linha: {e}"

# Botão de previsão para input individual
if st.sidebar.button("🔍 Prever Falhas (Individual)"):
    prediction = fazer_previsao({
        'Type': type_value,
        'Air temperature [K]': air_temp,
        'Process temperature [K]': process_temp,
        'Rotational speed [rpm]': rot_speed,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear
    }, 0)  # Linha 0 para individual

    # Exibindo o resultado
    if isinstance(prediction, list):
        if prediction:
            falhas = ', '.join(prediction)
            individual_pred_container.markdown(
                f"""
                <div style="padding:10px; border-radius:5px; background-color: #cb0000;">
                    <h3 style="text-align: center; color: white;">Falhas Previstas</h3>
                    <p style="text-align: center; font-size: 20px; font-weight: bold;">{falhas}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            individual_pred_container.markdown(
                f"""
                <div style="padding:10px; border-radius:5px; background-color: #26b500;">
                    <h3 style="text-align: center; color: white;">Sem Falhas Previstas</h3>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        individual_pred_container.error(prediction)

# Seção para processamento automático do CSV
st.header("📈 Processamento Automático de CSV")

# Placeholder para exibir o resultado em tempo real
auto_pred_container = st.empty()
progress_bar = st.progress(0)

# Botão para iniciar ou parar o processamento automático
if st.button("🚀 Iniciar Processamento Automático"):
    if st.session_state.processing:
        st.session_state.processing = False
        st.success("Processamento automático parado pelo usuário.")
    else:
        if st.session_state.current_index >= len(data):
            st.warning("Todos os dados já foram processados.")
        else:
            st.session_state.processing = True
            # Iniciar o processamento
            while st.session_state.processing and st.session_state.current_index < len(data):
                row = data.iloc[st.session_state.current_index]
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Fazer a predição
                prediction = fazer_previsao(row, st.session_state.current_index)
                # Atualizar as predições no session_state
                st.session_state.predictions.append({
                    'Linha': st.session_state.current_index + 1,
                    'Tempo': timestamp,
                    'Type': row['Type'],
                    'Air temperature [K]': row['Air temperature [K]'],
                    'Process temperature [K]': row['Process temperature [K]'],
                    'Rotational speed [rpm]': row['Rotational speed [rpm]'],
                    'Torque [Nm]': row['Torque [Nm]'],
                    'Tool wear [min]': row['Tool wear [min]'],
                    'Falhas_Previstas': ', '.join(prediction) if isinstance(prediction, list) and prediction else 'Sem Falhas'
                })
                # Atualizar o placeholder com a predição
                if isinstance(prediction, list):
                    if prediction:
                        falhas = ', '.join(prediction)
                        auto_pred_container.markdown(
                            f"""
                            <div style="margin: 10px; padding:10px; border-radius:25px; background-color: #cb0000; position: relative;">
                                <h4 style="text-align: center; color: white;">Falhas Previstas</h4>
                                <p style="text-align: center; font-size: 16px; font-weight: bold;">{falhas}</p>
                                <p style="font-size: 15px; font-weight: bold; position: absolute; bottom: 10px; right: 10px; margin: 0;">Instância: {st.session_state.current_index + 1}</p> 
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        auto_pred_container.markdown(
                            f"""
                            <div style="margin: 10px; padding:10px; border-radius:25px; background-color: #26b500; position: relative;">
                                <h4 style="text-align: center; color: white;">Sem Falhas Previstas</h4>
                                <p style="font-size: 15px; font-weight: bold; position: absolute; bottom: 10px; right: 10px; margin: 0;">Instância: {st.session_state.current_index + 1}</p> 
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    auto_pred_container.markdown(
                        f"""
                        <div style="margin: 10px; padding:10px; border-radius:25px; background-color: #ff9900; position: relative;">
                            <h4 style="text-align: center; color: white;">{prediction}</h4>
                            <p style="font-size: 15px; font-weight: bold; position: absolute; bottom: 10px; right: 10px; margin: 0;">Instância: {st.session_state.current_index + 1}</p> 
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                # Atualizar a barra de progresso
                progress = (st.session_state.current_index + 1) / len(data)
                progress_bar.progress(progress)
                # Incrementar o índice
                st.session_state.current_index += 1
                # Esperar 3 segundos
                time.sleep(3)
            if st.session_state.current_index >= len(data):
                st.session_state.processing = False
                st.success("Processamento automático concluído.")

# Botão para parar o processamento automático
if st.session_state.processing:
    if st.button("⏹ Parar Processamento Automático"):
        st.session_state.processing = False
        st.success("Processamento automático parado pelo usuário.")

# Exibir as predições realizadas
if st.session_state.predictions:
    st.header("📄 Predições Realizadas")
    predictions_df = pd.DataFrame(st.session_state.predictions)
    st.dataframe(predictions_df)

    # Adicionar gráficos de evolução das features
    st.subheader("📈 Evolução das Features ao Longo do Tempo")

    # Selecionar as colunas de features
    feature_columns = ['Type', 'Air temperature [K]', 'Process temperature [K]', 
                       'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

    # Criar um DataFrame apenas com as features e o tempo
    features_evolution = predictions_df[['Tempo'] + feature_columns].copy()
    features_evolution['Tempo'] = pd.to_datetime(features_evolution['Tempo'])

    # Ordenar por tempo
    features_evolution = features_evolution.sort_values('Tempo')

    # Configurar o layout dos gráficos