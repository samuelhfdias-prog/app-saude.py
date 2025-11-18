import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
import random
from datetime import datetime, timedelta
import numpy as np


# --- 1. Carregamento e Pré-processamento de Dados ---
# Se você tiver o arquivo 'saude_processada.csv', substitua este bloco
# pela linha: df = pd.read_csv('saude_processada.csv')
# E então, verifique se a coluna 'idade' e 'faixa_etaria' já existem.
# Se não existirem, adicione a lógica de cálculo após o pd.read_csv.

@st.cache_data  # Cache para otimizar o carregamento e pré-processamento
def load_and_preprocess_data():
    """
    Simula o carregamento e pré-processamento de dados da base saude_processada.csv,
    utilizando os atributos fornecidos:
    _id,sexo,cidade,bairro,dataNascimento,tipo,servico,dataEntrada,dataSaida,
    queixa,diagnostico,procedimento,descricaoMedicamento
    """
    num_registros = 1500  # Aumentado para uma simulação mais rica

    # Listas de valores possíveis para simulação
    sexos = ['Masculino', 'Feminino', 'Outro']
    cidades = ['São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Porto Alegre', 'Curitiba', 'Salvador']
    bairros = ['Centro', 'Jardins', 'Barra', 'Copacabana', 'Savassi', 'Cidade Baixa', 'Pinheiros', 'Lagoa']
    tipos_atendimento = ['Consulta', 'Emergência', 'Exame', 'Internação', 'Retorno']
    servicos = ['Clínica Geral', 'Pediatria', 'Cardiologia', 'Dermatologia', 'Ortopedia', 'Ginecologia']

    queixas_comuns = [
        'Dor de cabeça', 'Dor nas costas', 'Fadiga', 'Tosse', 'Dor de garganta',
        'Náusea', 'Febre', 'Azia', 'Dores musculares', 'Problemas de sono',
        'Alergia', 'Dificuldade para respirar', 'Dor no peito', 'Tontura', 'Ansiedade'
    ]
    diagnosticos_comuns = [
        'Resfriado Comum', 'Gripe', 'Infecção Urinária', 'Hipertensão Essencial',
        'Diabetes Mellitus Tipo 2', 'Gastrite Crônica', 'Enxaqueca',
        'Asma Brônquica', 'Dermatite Atópica', 'Ansiedade Generalizada',
        'Depressão Leve', 'Dor Lombar Inespecífica', 'Amigdalite Bacteriana',
        'Rinite Alérgica', 'Osteoartrite', 'Cistite'
    ]
    procedimentos_comuns = [
        'Consulta Médica', 'Exame de Sangue', 'Raio-X', 'Sutura', 'Aplicação de Medicamento',
        'Encaminhamento para Especialista', 'Aferição de Sinais Vitais', 'Curativo'
    ]
    medicamentos_comuns = [
        'Paracetamol', 'Dipirona', 'Ibuprofeno', 'Amoxicilina', 'Omeprazol',
        'Loratadina', 'Captopril', 'Metformina', 'Sinvastatina', 'Prednisona'
    ]

    data = {
        '_id': [f'rec_{i:06d}' for i in range(num_registros)],
        'sexo': [random.choice(sexos) for _ in range(num_registros)],
        'cidade': [random.choice(cidades) for _ in range(num_registros)],
        'bairro': [random.choice(bairros) for _ in range(num_registros)],
        'dataNascimento': [
            (datetime.now() - timedelta(days=random.randint(365 * 1, 365 * 90))).strftime('%Y-%m-%d')
            # Idade de 1 a 90 anos
            for _ in range(num_registros)
        ],
        'tipo': [random.choice(tipos_atendimento) for _ in range(num_registros)],
        'servico': [random.choice(servicos) for _ in range(num_registros)],
        'queixa': [random.choice(queixas_comuns) for _ in range(num_registros)],
        'diagnostico': [random.choice(diagnosticos_comuns) for _ in range(num_registros)],
        'procedimento': [random.choice(procedimentos_comuns) for _ in range(num_registros)],
        'descricaoMedicamento': [random.choice(medicamentos_comuns) for _ in range(num_registros)]
    }

    try:
        # Tenta carregar o CSV. Esta é a opção preferencial.
        df = pd.read_csv('saude_processada.csv')
        # O número de registros agora é o tamanho real do DataFrame
        num_registros = len(df)
    except FileNotFoundError:
        # Se o CSV não for encontrado, gera dados simulados.
        st.warning("Arquivo 'saude_processada.csv' não encontrado. Usando dados simulados.")
        df = pd.DataFrame(data)

    # Gerar dataEntrada e dataSaida (dataSaida sempre depois de dataEntrada)
    # A geração agora usa o número correto de registros (seja do CSV ou simulado)
    base_date = datetime(2023, 1, 1)  # Data base para entradas
    df['dataEntrada'] = [
        (base_date + timedelta(days=random.randint(0, 364), hours=random.randint(0, 23), minutes=random.randint(0, 59)))
        for _ in range(num_registros)
    ]
    df['dataSaida'] = df['dataEntrada'].apply(
        lambda x: x + timedelta(hours=random.randint(1, 48)))  # Saída entre 1 e 48h depois

    # Converter dataNascimento para datetime para calcular idade
    df['dataNascimento'] = pd.to_datetime(df['dataNascimento'])

    # Calcular idade e faixa etária
    # Usamos uma data de referência fixa para o cálculo da idade (e.g., 2024-01-01)
    # para garantir que a idade não mude a cada execução do app no mesmo dia.
    today = datetime(2024, 1, 1)
    df['idade'] = ((today - df['dataNascimento']).dt.days / 365.25).astype(int)

    bins = [0, 12, 18, 60, np.inf]  # Limites das faixas etárias
    labels = ['Criança', 'Adolescente', 'Adulto', 'Idoso']
    df['faixa_etaria'] = pd.cut(df['idade'], bins=bins, labels=labels, right=False, include_lowest=True)

    # Garantir que as colunas de texto sejam strings para a nuvem de palavras e gráficos
    df['queixa'] = df['queixa'].astype(str)
    df['diagnostico'] = df['diagnostico'].astype(str)

    return df


df = load_and_preprocess_data()

# --- Configurações da Página Streamlit ---
st.set_page_config(
    page_title="Análise de Queixas e Diagnósticos",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Título Principal ---
st.title("�� Dashboard de Análise de Queixas e Diagnósticos")
st.markdown("""
    Explore padrões e tendências nos dados de saúde utilizando filtros interativos e visualizações dinâmicas.
""")

# --- Sidebar para Filtros ---
st.sidebar.header("⚙️ Filtros de Análise")

# Opções de filtro para 'sexo'
all_sexos = ['Todos'] + sorted(df['sexo'].unique().tolist())
selected_sexo = st.sidebar.multiselect(
    "Selecione o Sexo:",
    options=all_sexos,
    default=['Todos']
)

# Opções de filtro para 'faixa_etaria'
# Usamos .dropna() para garantir que não haja NaNs ao pegar os únicos
all_faixas = ['Todas'] + sorted(df['faixa_etaria'].dropna().unique().tolist())
selected_faixa_etaria = st.sidebar.multiselect(
    "Selecione a Faixa Etária:",
    options=all_faixas,
    default=['Todas']
)

# --- Aplicação dos Filtros ---
filtered_df = df.copy()

if 'Todos' not in selected_sexo and selected_sexo:  # Verifica se 'Todos' não está selecionado e se há seleções
    filtered_df = filtered_df[filtered_df['sexo'].isin(selected_sexo)]

if 'Todas' not in selected_faixa_etaria and selected_faixa_etaria:  # Verifica se 'Todas' não está selecionado e se há seleções
    filtered_df = filtered_df[filtered_df['faixa_etaria'].isin(selected_faixa_etaria)]

# --- Verificação de dados após filtragem ---
if filtered_df.empty:
    st.warning("Nenhum dado encontrado para os filtros selecionados. Por favor, ajuste os filtros.")
else:
    # --- Gráficos Interativos (Plotly Express) ---
    st.header("📊 Principais Visualizações")
    col1, col2 = st.columns(2)  # Duas colunas para os gráficos

    with col1:
        st.subheader("Top 15 Diagnósticos Mais Frequentes")
        # Usando a coluna 'diagnostico' fornecida
        diagnosticos_counts = filtered_df['diagnostico'].value_counts().reset_index()
        diagnosticos_counts.columns = ['Diagnóstico', 'Frequência']
        diagnosticos_counts = diagnosticos_counts.sort_values(by='Frequência', ascending=False).head(15)

        if not diagnosticos_counts.empty:
            fig_diagnosticos = px.bar(
                diagnosticos_counts.sort_values(by='Frequência', ascending=True),  # Ordena para o menor ficar embaixo
                x='Frequência',
                y='Diagnóstico',
                orientation='h',
                labels={'Frequência': 'Número de Casos', 'Diagnóstico': ''},
                color='Frequência',
                color_continuous_scale=px.colors.sequential.Tealgrn,
                height=450
            )
            fig_diagnosticos.update_layout(showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_diagnosticos, use_container_width=True)
        else:
            st.info("Sem dados de diagnósticos para exibir com os filtros atuais.")

    with col2:
        st.subheader("Top 15 Queixas Mais Frequentes")
        # Usando a coluna 'queixa' fornecida
        queixas_counts = filtered_df['queixa'].value_counts().reset_index()
        queixas_counts.columns = ['Queixa', 'Frequência']
        queixas_counts = queixas_counts.sort_values(by='Frequência', ascending=False).head(15)

        if not queixas_counts.empty:
            fig_queixas = px.bar(
                queixas_counts.sort_values(by='Frequência', ascending=True),  # Ordena para o menor ficar embaixo
                x='Frequência',
                y='Queixa',
                orientation='h',
                labels={'Frequência': 'Número de Casos', 'Queixa': ''},
                color='Frequência',
                color_continuous_scale=px.colors.sequential.Oranges,
                height=450
            )
            fig_queixas.update_layout(showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_queixas, use_container_width=True)
        else:
            st.info("Sem dados de queixas para exibir com os filtros atuais.")

    # --- Nuvem de Palavras ---
    st.header("☁️ Nuvem de Palavras: Termos Relevantes")

    # Concatenar todos os diagnósticos e queixas em uma única string para a nuvem
    # Usando as colunas 'diagnostico' e 'queixa'
    text_diagnosticos = ' '.join(filtered_df['diagnostico'].dropna().tolist())
    text_queixas = ' '.join(filtered_df['queixa'].dropna().tolist())
    full_text = text_diagnosticos + ' ' + text_queixas

    # Lista de stopwords em português aprimorada
    stopwords = set([
        'a', 'ao', 'aos', 'aquela', 'aquelas', 'aquele', 'aqueles', 'aquilo', 'as', 'às', 'até', 'com', 'como', 'da',
        'das', 'de', 'dela', 'delas', 'dele', 'deles', 'depois', 'do', 'dos', 'e', 'é', 'ela', 'elas', 'ele', 'eles',
        'em', 'entre', 'era', 'eram', 'essa', 'essas', 'esse', 'esses', 'esta', 'está', 'estamos', 'estão', 'estas',
        'este', 'esteja', 'estejam', 'estejamos', 'estes', 'estive', 'estivemos', 'estiveram', 'estivermos',
        'estivesse',
        'estivessem', 'estivéssemos', 'estou', 'eu', 'foi', 'fomos', 'for', 'fora', 'foram', 'forem', 'formos', 'fosse',
        'fossem', 'fôssemos', 'fui', 'há', 'havia', 'hei', 'houve', 'houvemos', 'houver', 'houvera', 'houverá',
        'houveram',
        'houverão', 'houveria', 'houveriam', 'houveríamos', 'houvermos', 'houvesse', 'houvessem', 'houvéssemos', 'isso',
        'isto', 'já', 'lhe', 'lhes', 'mais', 'mas', 'me', 'mesmo', 'meu', 'meus', 'minha', 'minhas', 'muito', 'na',
        'não',
        'nas', 'nem', 'no', 'nos', 'nós', 'nossa', 'nossas', 'nosso', 'nossos', 'num', 'numa', 'o', 'os', 'ou', 'para',
        'pela', 'pelas', 'pelo', 'pelos', 'por', 'porque', 'qual', 'quando', 'que', 'quem', 'se', 'seja', 'sejam',
        'sejamos',
        'sem', 'ser', 'será', 'serão', 'seria', 'seriam', 'seríamos', 'seu', 'seus', 'só', 'somos', 'sou', 'sua',
        'suas',
        'também', 'te', 'tem', 'tém', 'temos', 'tenha', 'tenham', 'tenhamos', 'tenho', 'terá', 'terão', 'teria',
        'teriam',
        'teríamos', 'teu', 'teus', 'ti', 'tido', 'tinha', 'tinham', 'tínhamos', 'tive', 'tivemos', 'tiver', 'tivera',
        'tiveram', 'tivermos', 'tivesse', 'tivessem', 'tivéssemos', 'tu', 'tua', 'tuas', 'um', 'uma', 'uns', 'você',
        'vocês', 'vos', 'à', 'às', 'ó', 'já',
        # Termos genéricos para contexto médico/saúde que podem ser irrelevantes
        'tipo', 'crônica', 'severa', 'maior', 'recorrentes', 'generalizada', 'óssea', 'articular',
        'bacteriana', 'viral', 'extrema', 'constante', 'excessivo', 'inesperado', 'intensa',
        'aguda', 'leve', 'moderada', 'grave', 'sintomas', 'doença', 'paciente', 'histórico', 'diagnóstico',
        'infecção', 'inflamação', 'síndrome', 'distúrbio', 'crise', 'ataque', 'recorrência', 'agudo',
        'secundária', 'primária', 'cuidado', 'tratamento', 'terapia', 'medicamento', 'clínica', 'geral',
        'e', 'ou', 'por', 'que', 'se', 'ao', 'aos', 'à', 'às', 'no', 'na', 'nos', 'nas', 'um', 'uma', 'os', 'as'
    ])

    if full_text.strip():
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='cividis',  # Outro colormap para variar
            min_font_size=10,
            stopwords=stopwords,
            collocations=False,
            normalize_plurals=True
        ).generate(full_text)

        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)
    else:
        st.info("Sem termos para gerar a nuvem de palavras com os filtros atuais.")

# --- Informações adicionais na sidebar ---
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Sobre esta Aplicação:**
    Desenvolvido para transformar dados de saúde em insights visuais e acionáveis.
    Construído com Python, Streamlit, Plotly e WordCloud, utilizando os atributos
    específicos da base de dados fornecida.

    **Orvate - Consultor em Tecnologias Digitais**
    *Inovando para um Futuro Digital Mais Humano e Eficiente.*
    """
)