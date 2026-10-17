import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------------------------------
# Configuração da Página
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Previsão de Consumo Energético",
    page_icon="⚡",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 1. Geração do Dataset Sintético (Pandas / NumPy)
# -----------------------------------------------------------------------------
@st.cache_data
def carregar_dados():
    np.random.seed(42)
    n_registros = 50  # Mais de 20 registros conforme solicitado
    
    # Geração das variáveis preditoras
    velocidade = np.random.uniform(1.0, 5.0, n_registros)      # m/s
    maquinas = np.random.randint(1, 11, n_registros)           # contagem
    temp = np.random.uniform(18.0, 36.0, n_registros)          # °C
    
    # Relação linear básica com ruído para simular o consumo real
    # Consumo_base = 50 + 15*Velocidade + 25*Maquinas + 2*Temp + ruído
    ruido = np.random.normal(0, 5, n_registros)
    consumo = 50.0 + (15.5 * velocidade) + (24.8 * maquinas) + (2.1 * temp) + ruido

    df = pd.DataFrame({
        'Velocidade_Esteira': np.round(velocidade, 2),
        'Maquinas_Ativas': maquinas,
        'Temp_Ambiente': np.round(temp, 1),
        'Consumo_kWh': np.round(consumo, 2)
    })
    return df

df = carregar_dados()

# -----------------------------------------------------------------------------
# 2. Treinamento do Modelo de Regressão Linear (scikit-learn)
# -----------------------------------------------------------------------------
features = ['Velocidade_Esteira', 'Maquinas_Ativas', 'Temp_Ambiente']
target = 'Consumo_kWh'

X = df[features]
y = df[target]

modelo = LinearRegression()
modelo.fit(X, y)

# -----------------------------------------------------------------------------
# 3. Barra Lateral (Sidebar): Parâmetros de Entrada
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Parâmetros Operacionais")
st.sidebar.markdown("Ajuste as variáveis de processo para simular o consumo:")

input_velocidade = st.sidebar.number_input(
    label="Velocidade da Esteira (m/s):",
    min_value=0.5,
    max_value=10.0,
    value=3.0,
    step=0.1
)

input_maquinas = st.sidebar.number_input(
    label="Máquinas Ativas (unidades):",
    min_value=1,
    max_value=20,
    value=5,
    step=1
)

input_temp = st.sidebar.number_input(
    label="Temperatura Ambiente (°C):",
    min_value=10.0,
    max_value=50.0,
    value=25.0,
    step=0.5
)

# -----------------------------------------------------------------------------
# 4. Tela Principal: Previsão e Métricas
# -----------------------------------------------------------------------------
st.title("⚡ Painel de Previsão de Consumo de Energia Industrial")
st.markdown("Monitoramento preditivo de consumo elétrico utilizando **Regressão Linear**.")

# Dados de entrada formatados para predição
dados_usuario = pd.DataFrame([{
    'Velocidade_Esteira': input_velocidade,
    'Maquinas_Ativas': input_maquinas,
    'Temp_Ambiente': input_temp
}])

# Realizando a predição
predicao = modelo.predict(dados_usuario)[0]

# Exibição do resultado em destaque
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Resultado Estimado")
    st.metric(
        label="Consumo Previsto",
        value=f"{predicao:.2f} kWh",
        delta=f"{(predicao - df['Consumo_kWh'].mean()):+.2f} kWh vs média histórica",
        delta_color="inverse"
    )
    
    st.markdown("### Resumo das Entradas:")
    st.write(f"- **Velocidade:** `{input_velocidade} m/s`")
    st.write(f"- **Máquinas Ativas:** `{input_maquinas}`")
    st.write(f"- **Temperatura:** `{input_temp} °C`")

# -----------------------------------------------------------------------------
# 5. Visualização: Contribuição das Variáveis (Matplotlib)
# -----------------------------------------------------------------------------
with col2:
    st.subheader("Importância das Variáveis no Modelo")
    
    # Coeficientes da regressão linear
    coeficientes = modelo.coef_
    nomes_colunas = ['Velocidade da Esteira', 'Máquinas Ativas', 'Temp. Ambiente']

    fig, ax = plt.subplots(figsize=(7, 4.5))
    barras = ax.barh(nomes_colunas, coeficientes, color=['#2b5c8f', '#d95f02', '#7570b3'], edgecolor='black')
    
    ax.set_xlabel("Impacto no Consumo por Unidade (kWh)")
    ax.set_title("Coeficientes da Regressão Linear (Pesos)")
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    # Rótulo com o valor em cada barra
    for barra in barras:
        largura = barra.get_width()
        ax.annotate(f"{largura:.2f}",
                    xy=(largura, barra.get_y() + barra.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center',
                    fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)

# -----------------------------------------------------------------------------
# 6. Tabela com Dados Históricos
# -----------------------------------------------------------------------------
with st.expander("🔍 Visualizar Dataset Histórico de Treinamento (Primeiras 10 linhas)"):
    st.dataframe(df.head(10), use_container_width=True)
