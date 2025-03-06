import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

# 1. Carregar os dados do arquivo Excel
file_path = 'base_e_calculadora_cafe.xlsx'

# Carregar a aba "Base" para treinar o modelo
base_df = pd.read_excel(file_path, sheet_name='Base')

# 2. Preparar os dados para o modelo de regressão logística
# Converter variáveis categóricas em variáveis dummy
X = pd.get_dummies(base_df[['Pais_Origem', 'Tipo_Variedade', 'Tipo_Processamento', 'Faixa de Altitude']], drop_first=True)
y = base_df['Tipo_Qualidade']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Treinar o modelo de regressão logística
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 4. Avaliar o modelo
y_pred = model.predict(X_test)
print("Acurácia do modelo:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

# 5. Função para prever a probabilidade de ser Premium
def predict_probability(pais_origem, tipo_variedade, tipo_processamento, faixa_altitude):
    # Criar um DataFrame com os dados de entrada
    input_data = pd.DataFrame({
        'Pais_Origem': [pais_origem],
        'Tipo_Variedade': [tipo_variedade],
        'Tipo_Processamento': [tipo_processamento],
        'Faixa de Altitude': [faixa_altitude]
    })
    
    # Converter variáveis categóricas em variáveis dummy
    input_data = pd.get_dummies(input_data, drop_first=True)
    
    # Garantir que as colunas estejam na mesma ordem que o modelo foi treinado
    input_data = input_data.reindex(columns=X.columns, fill_value=0)
    
    # Fazer a previsão da probabilidade de ser Premium
    probability = model.predict_proba(input_data)[0][1]  # Probabilidade de ser Premium
    
    return probability

# 6. Interface do Streamlit
st.title("Calculadora de Garantia de Café")

# Inputs do usuário
st.sidebar.header("Dados do Café")
pais_origem = st.sidebar.selectbox("País de Origem", base_df['Pais_Origem'].unique())
tipo_variedade = st.sidebar.selectbox("Tipo de Variedade", base_df['Tipo_Variedade'].unique())
tipo_processamento = st.sidebar.selectbox("Tipo de Processamento", base_df['Tipo_Processamento'].unique())
faixa_altitude = st.sidebar.selectbox("Faixa de Altitude", base_df['Faixa de Altitude'].unique())
quantidade_sacas = st.sidebar.number_input("Quantidade de Sacas de 60kg", min_value=1, value=1000)

# Valores fixos
valor_saca_premium = 1800  # R$ 1.800 por saca de 60kg (Premium)
valor_saca_normal = 800    # R$ 800 por saca de 60kg (Normal)

# Botão para calcular
if st.sidebar.button("Calcular Garantia"):
    # Calcular a probabilidade de ser Premium
    prob_premium = predict_probability(pais_origem, tipo_variedade, tipo_processamento, faixa_altitude)
    
    # Calcular o Valor Aproximado da Garantia
    valor_garantia = (prob_premium * valor_saca_premium + (1 - prob_premium) * valor_saca_normal) * quantidade_sacas
    
    # Calcular o Valor Máximo a ser Liberado (70% do Valor da Garantia)
    valor_maximo_liberado = 0.7 * valor_garantia
    
    # Exibir os resultados
    st.write(f"### Resultados:")
    st.write(f"**Probabilidade de ser Premium:** {prob_premium:.2%}")
    st.write(f"**Valor Aproximado da Garantia:** R$ {valor_garantia:,.2f}")
    st.write(f"**Valor Máximo a ser Liberado:** R$ {valor_maximo_liberado:,.2f}")
