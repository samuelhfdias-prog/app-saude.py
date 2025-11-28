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
    page_title="Analytics Saúde | Dashboard Executivo",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CAMADA DE ESTILO (CSS - UI/UX) ---
def aplicar_estilo_css():
    st.markdown("""
        <style>
        /* Fundo e Fonte Global */
        .stApp {
            background-color: #0e1117;
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }
        
        /* Container Personalizado (Card Effect) */
        .css-card {
            background-color: #1e2130;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            border: 1px solid #30334e;
            margin-bottom: 20px;
        }
        
        /* Melhoria nos KPIs (Métricas) */
        div[data-testid="stMetric"] {
            background-color: #1e2130;
            border: 1px solid #30334e;
            padding: 15px;
            border-radius: 8px;
            text-align: center; 
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.85rem !important;
            color: #b0b3c5 !important;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            color: #4db8ff !important; /* Azul médico vibrante */
        }
        
        /* Títulos de Seção */
        h1, h2, h3 {
            color: #fafafa;
            font-weight: 600;
        }
        
        /* Abas (Tabs) */
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
            border-bottom: 1px solid #30334e;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 0px;
            color: #b0b3c5;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            color: #4db8ff !important;
            border-bottom: 2px solid #4db8ff;
            background-color: transparent !important;
        }
        </style>
    """, unsafe_allow_html=True)

# --- FUNÇÕES UTILITÁRIAS (Lógica Mantida) ---

def remover_acentos(texto):
    if not isinstance(texto, str):
        return str(texto)
    nfkd = unicodedata.normalize('NFKD', texto)
    return u"".join([c for c in nfkd if not unicodedata.combining(c)])

# --- 1. CARREGAMENTO E PREPARAÇÃO DE DADOS (Lógica Mantida) ---

@st.cache_data
def gerar_dados_simulados(num_registros=1500):
    sexos = ['Masculino', 'Feminino', 'Outro', np.nan]
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
    df = df_input.copy()
    cols_texto = ['sexo', 'cidade', 'bairro', 'queixa', 'diagnostico', 'tipo', 'servico']
    for col in cols_texto:
        if col in df.columns:
            # Garante que 'Não Informado' seja aplicado explicitamente para visualização
            df[col] = df[col].fillna('Não Informado').astype(str)
            df[col] = df[col].replace(['nan', 'NaN', 'None', ''], 'Não Informado')

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
    try:
        df_raw = pd.read_csv('saude_processada.csv')
    except FileNotFoundError:
        df_raw = gerar_dados_simulados()
    return preparar_base(df_raw)

# --- 2. COMPONENTES VISUAIS (Atualizado para UI/UX) ---

def layout_kpis(df):
    """Gera KPIs estilizados, permitindo visualização de dados não informados."""
    total_pacientes = len(df)
    media_idade = df['idade'].mean() if not df.empty else 0
    
    top_cidade = "Sem Dados"
    if not df.empty:
        moda_cidade = df['cidade'].mode()
        if not moda_cidade.empty:
            top_cidade = moda_cidade[0]

    top_diag = "Inconclusivo"
    if not df.empty:
        # Alteração: Permite que 'Não Informado' apareça como Top Diagnóstico se for a moda
        top_diag = df['diagnostico'].mode()[0]

    # Layout de 4 colunas para KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🏥 Total Atendimentos", f"{total_pacientes:,}".replace(",", "."), delta_color="off")
    k2.metric("🎂 Idade Média", f"{media_idade:.1f} anos")
    k3.metric("📍 Cidade Principal", top_cidade)
    k4.metric("🦠 Top Diagnóstico", top_diag)

def plot_barra_horizontal(df, x_col, y_col, titulo, cor_seq):
    """Função auxiliar para gráficos limpos do Plotly."""
    fig = px.bar(
        df, x=x_col, y=y_col, orientation='h', text=x_col,
        title=titulo, color=x_col, color_continuous_scale=cor_seq
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        yaxis=dict(autorange="reversed", title=None),
        xaxis=dict(showgrid=False, title=None),
        margin=dict(l=0, r=0, t=40, b=0),
        height=350,
        showlegend=False
    )
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

def graficos_demograficos(df):
    """Novos gráficos para enriquecer a análise (Sexo e Distribuição de Idade)."""
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("### 🚻 Distribuição por Gênero")
        if df.empty:
            st.info("Sem dados.")
        else:
            df_sexo = df['sexo'].value_counts().reset_index()
            df_sexo.columns = ['Sexo', 'Total']
            fig = px.pie(df_sexo, values='Total', names='Sexo', hole=0.5, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#fff'), height=300)
            st.plotly_chart(fig, use_container_width=True)
            
    with c2:
        st.markdown("### 🎂 Distribuição Etária")
        if df.empty:
            st.info("Sem dados.")
        else:
            fig = px.histogram(df, x="idade", nbins=15, title="", color_discrete_sequence=['#4db8ff'])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                font=dict(color='#fff'), bargap=0.1, height=300,
                xaxis=dict(showgrid=False, title="Idade"), yaxis=dict(showgrid=True, gridcolor='#30334e')
            )
            st.plotly_chart(fig, use_container_width=True)

def grafico_linha_tempo(df):
    """Novo gráfico de evolução temporal (Time Series)."""
    st.markdown("### 📈 Evolução dos Atendimentos (Diário)")
    if df.empty:
        st.info("Sem dados temporais.")
        return
    
    # Agrupamento por data
    df_tempo = df.groupby(df['dataEntrada'].dt.date).size().reset_index(name='Atendimentos')
    df_tempo.columns = ['Data', 'Atendimentos']
    
    fig = px.area(df_tempo, x='Data', y='Atendimentos', markers=True)
    fig.update_traces(line_color='#00d4aa', fillcolor='rgba(0, 212, 170, 0.2)')
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#fff'),
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#30334e'),
        height=350, margin=dict(l=0, r=0, t=10, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

def nuvem_termos_otimizada(df):
    if df.empty:
        st.warning("Sem dados.")
        return

    text = ' '.join(df['diagnostico'].astype(str) + ' ' + df['queixa'].astype(str))
    text = remover_acentos(text.lower())
    
    # Alteração: 'nao informado' e 'nao definido' removidos das stopwords para aparecerem na nuvem
    stopwords = set(['nan', 'dor', 'paciente', 'de', 'do', 'da'])

    wordcloud = WordCloud(
        width=800, height=350,
        background_color='#1e2130', # Combina com o card
        colormap='GnBu', # Cores Azul/Verde médico
        stopwords=stopwords,
        regexp=r"\w[\w']+"
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_alpha(0)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# --- 3. EXECUÇÃO PRINCIPAL ---

def main():
    aplicar_estilo_css()
    
    # --- Sidebar ---
    st.sidebar.title("⚕️ Filtros")
    df = carregar_dados()

    # Filtros mais compactos
    with st.sidebar.expander("🌍 Localização", expanded=True):
        sel_cidade = st.multiselect("Município:", sorted(df['cidade'].unique()), default=[])
    
    with st.sidebar.expander("👤 Perfil do Paciente", expanded=False):
        sel_sexo = st.multiselect("Sexo:", sorted(df['sexo'].unique()))
        sel_faixa = st.multiselect("Faixa Etária:", sorted(df['faixa_etaria'].unique()))

    # Aplicação dos Filtros
    df_filtrado = df.copy()
    if sel_cidade: df_filtrado = df_filtrado[df_filtrado['cidade'].isin(sel_cidade)]
    if sel_sexo: df_filtrado = df_filtrado[df_filtrado['sexo'].isin(sel_sexo)]
    if sel_faixa: df_filtrado = df_filtrado[df_filtrado['faixa_etaria'].isin(sel_faixa)]

    if df_filtrado.empty:
        st.error("Nenhum dado encontrado para os filtros selecionados.")
        return

    # --- Header Principal ---
    st.title("Monitoramento de Saúde Pública")
    st.markdown(f"Visão geral dos **{len(df_filtrado)} registros** filtrados na base de dados.")
    st.markdown("---")

    # --- SEÇÃO 1: KPIs ---
    layout_kpis(df_filtrado)
    st.markdown("<br>", unsafe_allow_html=True) # Espaçamento

    # --- SEÇÃO 2: Gráficos Principais (Tabs) ---
    tab1, tab2, tab3 = st.tabs(["📊 Visão Clínica", "👥 Demografia & Tempo", "📂 Dados Brutos"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🩺 Diagnósticos Mais Frequentes")
            counts = df_filtrado['diagnostico'].value_counts().head(10).reset_index()
            counts.columns = ['Diagnóstico', 'Qtd']
            plot_barra_horizontal(counts, 'Qtd', 'Diagnóstico', '', px.colors.sequential.Teal)
        
        with c2:
            st.markdown("### 🤕 Queixas Principais")
            counts = df_filtrado['queixa'].value_counts().head(10).reset_index()
            counts.columns = ['Queixa', 'Qtd']
            plot_barra_horizontal(counts, 'Qtd', 'Queixa', '', px.colors.sequential.Oranges)
        
        st.markdown("---")
        st.markdown("### ☁️ Nuvem de Sintomas e Diagnósticos")
        nuvem_termos_otimizada(df_filtrado)

    with tab2:
        grafico_linha_tempo(df_filtrado)
        st.markdown("---")
        graficos_demograficos(df_filtrado)

    with tab3:
        st.dataframe(
            df_filtrado[['dataEntrada', 'cidade', 'sexo', 'idade', 'queixa', 'diagnostico']],
            use_container_width=True,
            hide_index=True
        )

    # Footer simples
    st.sidebar.markdown("---")
    st.sidebar.info("Desenvolvido para análise epidemiológica.")

if __name__ == "__main__":
    main()    main()

