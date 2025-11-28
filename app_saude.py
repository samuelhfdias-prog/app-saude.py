import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta
import numpy as np
import unicodedata

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Dashboard de Saúde | Análise Clínica",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CAMADA DE ESTILO (CSS) ---
def aplicar_estilo_css():
    st.markdown("""
        <style>
        /* Ajuste global para tema escuro e contraste */
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        
        /* Estilização dos Cards (KPIs) */
        div[data-testid="stMetric"] {
            background-color: #262730;
            border: 1px solid #41424C;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            text-align: center;
            transition: transform 0.2s;
        }
        div[data-testid="stMetric"]:hover {
            transform: scale(1.02);
            border-color: #ff4b4b;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.9rem;
            color: #a3a8b8;
            font-weight: 500;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.6rem;
            color: #ffffff;
            font-weight: 700;
        }
        
        /* Responsividade para Telas Pequenas */
        @media (max-width: 768px) {
            div[data-testid="stMetric"] {
                margin-bottom: 10px;
            }
            .block-container {
                padding-top: 2rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
        
        /* Ajuste de Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #262730;
            border-radius: 5px;
            color: #fff;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ff4b4b !important;
        }
        </style>
    """, unsafe_allow_html=True)

# --- FUNÇÕES UTILITÁRIAS ---

def remover_acentos(texto):
    """Normaliza strings removendo acentos e caracteres especiais."""
    if not isinstance(texto, str):
        return str(texto)
    nfkd = unicodedata.normalize('NFKD', texto)
    return u"".join([c for c in nfkd if not unicodedata.combining(c)])

# --- 1. CARREGAMENTO E PREPARAÇÃO DE DADOS ---

@st.cache_data
def gerar_dados_simulados(num_registros=1500):
    """Gera dados simulados caso o CSV não exista."""
    sexos = ['Masculino', 'Feminino', 'Outro', np.nan] # Adicionado NaN para teste
    cidades = ['São Paulo', 'Pompeia', 'Belo Horizonte', 'Porto Alegre', 'Curitiba', 'Salvador']
    bairros = ['Centro', 'Jardins', 'Barra', 'Copacabana', 'Savassi', 'Industrial', 'Vila Nova']
    tipos_atendimento = ['Consulta', 'Emergência', 'Exame', 'Internação', 'Retorno']
    servicos = ['Clínica Geral', 'Pediatria', 'Cardiologia', 'Dermatologia', 'Ortopedia', 'Ginecologia']

    queixas_comuns = [
        'Dor de cabeça', 'Dor nas costas', 'Fadiga', 'Tosse', 'Febre', 
        'Náusea', 'Dores musculares', 'Ansiedade', 'Dor no peito', np.nan
    ]
    diagnosticos_comuns = [
        'Gripe', 'Infecção Urinária', 'Hipertensão', 'Diabetes Tipo 2', 
        'Gastrite', 'Enxaqueca', 'Asma', 'Dermatite', 'Ansiedade', 
        'Depressão', 'Não Definido', np.nan
    ]

    base_date = datetime(2023, 1, 1)
    
    data = {
        '_id': [f'rec_{i:06d}' for i in range(num_registros)],
        'sexo': [random.choice(sexos) for _ in range(num_registros)],
        'cidade': [random.choice(cidades) for _ in range(num_registros)],
        'bairro': [random.choice(bairros) for _ in range(num_registros)],
        'dataNascimento': [(datetime.now() - timedelta(days=random.randint(365*1, 365*90))).strftime('%Y-%m-%d') for _ in range(num_registros)],
        'tipo': [random.choice(tipos_atendimento) for _ in range(num_registros)],
        'servico': [random.choice(servicos) for _ in range(num_registros)],
        'queixa': [random.choice(queixas_comuns) for _ in range(num_registros)],
        'diagnostico': [random.choice(diagnosticos_comuns) for _ in range(num_registros)],
        'dataEntrada': [(base_date + timedelta(days=random.randint(0, 364), hours=random.randint(0, 23))) for _ in range(num_registros)]
    }
    
    df = pd.DataFrame(data)
    df['dataSaida'] = df['dataEntrada'].apply(lambda x: x + timedelta(hours=random.randint(1, 48)))
    return df

@st.cache_data
def preparar_base(df_input):
    """
    Realiza o tratamento de nulos, cálculo de idade e padronização.
    """
    df = df_input.copy()

    # 1. Tratamento de Nulos (Crítico para UX)
    cols_texto = ['sexo', 'cidade', 'bairro', 'queixa', 'diagnostico', 'tipo', 'servico']
    for col in cols_texto:
        if col in df.columns:
            # Preenche NaN com 'Não Informado' e converte para string
            df[col] = df[col].fillna('Não Informado').astype(str)
            # Padroniza variações como 'não definido' para 'Não Informado' se desejar unificar
            df[col] = df[col].replace(['nan', 'NaN', 'None', ''], 'Não Informado')

    # 2. Conversão de Datas e Idade
    if 'dataNascimento' in df.columns:
        df['dataNascimento'] = pd.to_datetime(df['dataNascimento'], errors='coerce')
        today = datetime(2024, 1, 1)
        df['idade'] = ((today - df['dataNascimento']).dt.days / 365.25).fillna(0).astype(int)
        
        bins = [0, 12, 18, 60, np.inf]
        labels = ['Criança', 'Adolescente', 'Adulto', 'Idoso']
        df['faixa_etaria'] = pd.cut(df['idade'], bins=bins, labels=labels, right=False, include_lowest=True)
        df['faixa_etaria'] = df['faixa_etaria'].astype(str).replace('nan', 'Não Informado')

    return df

@st.cache_data
def carregar_dados():
    """Tenta carregar CSV ou gera simulado, depois pré-processa."""
    try:
        df_raw = pd.read_csv('saude_processada.csv')
    except FileNotFoundError:
        df_raw = gerar_dados_simulados()
    
    return preparar_base(df_raw)

# --- 2. COMPONENTES VISUAIS ---

def criar_cards_resumo(df):
    """Gera os KPIs principais com tratamento de erros."""
    st.subheader("📌 Indicadores Chave")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    # Cálculos seguros
    total_pacientes = len(df)
    
    media_idade = df['idade'].mean() if not df.empty else 0
    
    top_cidade = "Sem Dados"
    if not df.empty:
        moda_cidade = df['cidade'].mode()
        if not moda_cidade.empty:
            top_cidade = moda_cidade[0]

    top_diag = "Sem Dados"
    if not df.empty:
        # Filtra 'Não Informado' para pegar o diagnóstico real mais comum
        diag_validos = df[~df['diagnostico'].isin(['Não Informado', 'Não Definido'])]
        if not diag_validos.empty:
            top_diag = diag_validos['diagnostico'].mode()[0]
        else:
            top_diag = "Inconclusivo"

    with kpi1: st.metric("Total de Atendimentos", f"{total_pacientes:,}".replace(",", "."))
    with kpi2: st.metric("Média de Idade", f"{media_idade:.1f} anos")
    with kpi3: st.metric("Cidade + Frequente", top_cidade)
    with kpi4: st.metric("Principal Diagnóstico", top_diag, help="Exclui 'Não Informado'")

def _gerar_grafico_barras(df, coluna, titulo, cor_escala):
    """Função genérica reutilizável para gráficos de barra."""
    if df.empty:
        st.info(f"Sem dados para {titulo}.")
        return

    # Contagem e ordenação
    counts = df[coluna].value_counts().reset_index()
    counts.columns = [coluna, 'Frequência']
    counts = counts.sort_values(by='Frequência', ascending=False).head(15) # Top 15
    
    # Destacar "Não Informado" visualmente?
    # Aqui optamos por mantê-lo mas ordenado. O Plotly lida bem com cores.

    fig = px.bar(
        counts,
        x='Frequência',
        y=coluna,
        orientation='h',
        text='Frequência',
        color='Frequência',
        color_continuous_scale=cor_escala,
        title=titulo
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#fafafa'),
        yaxis=dict(autorange="reversed", title=""), # Maior no topo
        xaxis=dict(title="Número de Casos"),
        margin=dict(l=10, r=20, t=40, b=10),
        height=400
    )
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

def grafico_top_diagnosticos(df):
    _gerar_grafico_barras(df, 'diagnostico', 'Top 15 Diagnósticos', px.colors.sequential.Tealgrn)

def grafico_top_queixas(df):
    _gerar_grafico_barras(df, 'queixa', 'Top 15 Queixas Principais', px.colors.sequential.Oranges)

def nuvem_termos(df):
    """Gera nuvem de palavras com limpeza avançada."""
    st.markdown("### ☁️ Nuvem de Termos Relevantes")
    
    if df.empty:
        st.warning("Sem dados para gerar a nuvem.")
        return

    # 1. Concatenação
    text_diagnosticos = ' '.join(df['diagnostico'].tolist())
    text_queixas = ' '.join(df['queixa'].tolist())
    full_text = text_diagnosticos + ' ' + text_queixas

    # 2. Limpeza Prévia
    full_text = remover_acentos(full_text.lower())

    # 3. Stopwords Customizadas (incluindo variações de nulos)
    stopwords = set([
        'nao informado', 'nao definido', 'nan', 'null', 'paciente', 'dor', 'de', 'da', 'do', 'em', 'para', 
        'com', 'que', 'e', 'ou', 'a', 'o', 'as', 'os', 'um', 'uma', 'uns', 'umas', 'nos', 'nas', 
        'cronica', 'aguda', 'leve', 'grave', 'sintomas', 'geral', 'tipo'
    ])

    if not full_text.strip():
        st.info("Texto insuficiente após limpeza.")
        return

    with st.spinner('Gerando nuvem de palavras...'):
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='#0e1117', # Fundo escuro para combinar com tema
            mode="RGBA",
            colormap='cool', # Cores vibrantes para fundo escuro
            min_font_size=12,
            stopwords=stopwords,
            collocations=False, # Evita duplicar palavras compostas simples
            regexp=r"\w[\w']+"
        ).generate(full_text)

        fig, ax = plt.subplots(figsize=(10, 5))
        # Fundo transparente no Matplotlib
        fig.patch.set_alpha(0) 
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

# --- 3. EXECUÇÃO PRINCIPAL ---

def main():
    aplicar_estilo_css()
    
    # Header
    col_img, col_txt = st.columns([0.5, 4.5])
    with col_txt:
        st.title("Monitoramento de Saúde Pública")
        st.markdown("Análise interativa de diagnósticos, queixas e perfil demográfico.")
    
    # Carregamento
    df = carregar_dados()

    # --- Sidebar (Filtros Eficientes) ---
    st.sidebar.header("🔎 Filtros Globais")
    
    # Filtro Cidade (Atendendo à solicitação de Pompeia)
    cidades_disp = sorted(df['cidade'].unique().tolist())
    sel_cidade = st.sidebar.multiselect("Município:", cidades_disp, default=[]) # Default vazio = todos
    
    # Filtro Sexo
    sexos_disp = sorted(df['sexo'].unique().tolist())
    sel_sexo = st.sidebar.multiselect("Sexo Biológico:", sexos_disp)

    # Filtro Faixa Etária
    faixas_disp = sorted(df['faixa_etaria'].unique().tolist())
    sel_faixa = st.sidebar.multiselect("Faixa Etária:", faixas_disp)

    # Aplicação dos Filtros (Lógica)
    df_filtrado = df.copy()
    
    if sel_cidade:
        df_filtrado = df_filtrado[df_filtrado['cidade'].isin(sel_cidade)]
    if sel_sexo:
        df_filtrado = df_filtrado[df_filtrado['sexo'].isin(sel_sexo)]
    if sel_faixa:
        df_filtrado = df_filtrado[df_filtrado['faixa_etaria'].isin(sel_faixa)]

    # Feedback de filtros vazios
    if df_filtrado.empty:
        st.warning("⚠️ Nenhum registro encontrado para os filtros selecionados.")
        return

    # --- Renderização do Conteúdo ---
    
    # 1. Cards (Indicadores)
    criar_cards_resumo(df_filtrado)
    
    st.divider()

    # 2. Tabs para Organização
    tab1, tab2, tab3 = st.tabs(["📊 Análise Gráfica", "☁️ Padrões Textuais", "📂 Dados Detalhados"])

    with tab1:
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            grafico_top_diagnosticos(df_filtrado)
        with col_g2:
            grafico_top_queixas(df_filtrado)

    with tab2:
        nuvem_termos(df_filtrado)

    with tab3:
        st.markdown(f"### Base de Dados Filtrada ({len(df_filtrado)} registros)")
        st.dataframe(
            df_filtrado,
            use_container_width=True,
            column_config={
                "dataNascimento": st.column_config.DateColumn("Data Nasc."),
                "idade": st.column_config.NumberColumn("Idade", format="%d anos"),
            },
            hide_index=True
        )

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2025 Orvate Tech")

if __name__ == "__main__":
    main()
