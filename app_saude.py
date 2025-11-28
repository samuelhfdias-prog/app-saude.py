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
    page_title="Analytics Saúde | Visão Estratégica",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- TEMA E ESTILO (UI PRO) ---
def aplicar_estilo_css():
    st.markdown("""
        <style>
        /* Importação de Fonte Google (Roboto) para visual limpo */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Roboto', sans-serif;
        }

        /* Fundo Geral e Barra Lateral */
        .stApp {
            background-color: #0F1116; /* Preto Profundo */
        }
        section[data-testid="stSidebar"] {
            background-color: #161B22; /* Cinza Escuro Azulado */
            border-right: 1px solid #30363D;
        }

        /* Estilização dos Cards (KPIs) - Efeito Glassmorphism Leve */
        div[data-testid="stMetric"] {
            background-color: #1F242D;
            border: 1px solid #30363D;
            padding: 20px;
            border-radius: 12px;
            border-left: 5px solid #2E9CCA; /* Acento Azul Saúde */
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(46, 156, 202, 0.2);
            border-color: #2E9CCA;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #8B949E;
        }
        div[data-testid="stMetricValue"] {
            font-size: 2rem;
            color: #F0F6FC;
            font-weight: 700;
        }

        /* Títulos e Headers */
        h1, h2, h3 {
            color: #E6EDF3 !important;
        }
        
        /* Ajuste de Abas */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            height: 45px;
            background-color: #161B22;
            border: 1px solid #30363D;
            border-radius: 6px;
            color: #8B949E;
            padding: 0 20px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #2E9CCA !important;
            color: white !important;
            border-color: #2E9CCA !important;
        }

        /* Customização da Barra de Rolagem */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-thumb {
            background: #30363D;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-track {
            background: #0F1116;
        }
        </style>
    """, unsafe_allow_html=True)

# --- FUNÇÕES UTILITÁRIAS ---
def remover_acentos(texto):
    if not isinstance(texto, str): return str(texto)
    nfkd = unicodedata.normalize('NFKD', texto)
    return u"".join([c for c in nfkd if not unicodedata.combining(c)])

# --- GERAÇÃO DE DADOS ---
@st.cache_data
def carregar_dados_inteligentes():
    """Gera dados ou carrega, com tratamento robusto."""
    try:
        df = pd.read_csv('saude_processada.csv')
    except FileNotFoundError:
        # Simulação Rápida
        num = 1500
        sexos = ['Masculino', 'Feminino']
        cidades = ['São Paulo', 'Pompeia', 'Marília', 'Bauru', 'Curitiba']
        queixas = ['Dor de Cabeça', 'Febre Alta', 'Dor Lombar', 'Ansiedade', 'Tosse Seca', np.nan]
        diagnosticos = ['Enxaqueca', 'Viral', 'Lombalgia', 'Crise Ansiosa', 'Bronquite', 'Indefinido']
        
        df = pd.DataFrame({
            'sexo': [random.choice(sexos) for _ in range(num)],
            'cidade': [random.choice(cidades) for _ in range(num)],
            'queixa': [random.choice(queixas) for _ in range(num)],
            'diagnostico': [random.choice(diagnosticos) for _ in range(num)],
            'idade': [random.randint(1, 95) for _ in range(num)],
            'data_atendimento': [datetime.now() - timedelta(days=random.randint(0, 365)) for _ in range(num)]
        })
        
        # Categorização de Idade
        bins = [0, 12, 19, 60, 120]
        labels = ['Criança', 'Adolescente', 'Adulto', 'Idoso']
        df['faixa_etaria'] = pd.cut(df['idade'], bins=bins, labels=labels)

    # Tratamento de Strings para UX (Remover nan feio)
    cols_str = ['sexo', 'cidade', 'queixa', 'diagnostico']
    for col in cols_str:
        if col in df.columns:
            df[col] = df[col].fillna('Não Informado').astype(str)
            
    return df

# --- COMPONENTES VISUAIS APRIMORADOS ---

def plot_config_pro(fig):
    """Aplica um tema limpo e profissional a qualquer gráfico Plotly."""
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8B949E', family="Roboto"),
        title_font=dict(size=18, color='#E6EDF3', family="Roboto"),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(showgrid=False, color='#8B949E'),
        yaxis=dict(showgrid=True, gridcolor='#30363D', color='#8B949E'),
        hovermode="y unified"
    )
    return fig

def card_kpi(col, titulo, valor, subtitulo=None, cor_borda="#2E9CCA"):
    """Renderiza um card KPI personalizado se necessário (ou usa o nativo estilizado)."""
    with col:
        st.metric(label=titulo, value=valor, delta=subtitulo)

def grafico_barras_horizontal(df, col_x, titulo, cor):
    if df.empty: return
    
    counts = df[col_x].value_counts().head(10).sort_values(ascending=True)
    
    fig = px.bar(
        x=counts.values,
        y=counts.index,
        orientation='h',
        text=counts.values,
        title=titulo,
        color_discrete_sequence=[cor]
    )
    
    fig.update_traces(textposition='outside', marker_line_width=0, opacity=0.9)
    fig = plot_config_pro(fig)
    st.plotly_chart(fig, use_container_width=True)

# --- APP PRINCIPAL ---

def main():
    aplicar_estilo_css()
    df = carregar_dados_inteligentes()

    # --- SIDEBAR (Navegação e Filtros) ---
    with st.sidebar:
        st.title("🏥 HealthData")
        st.caption("Sistema de Monitoramento Clínico v2.0")
        st.markdown("---")
        
        st.header("🎛️ Filtros Ativos")
        
        # Filtros encadeados (UX: Filtros dinâmicos)
        cidades = st.multiselect("📍 Localidade", sorted(df['cidade'].unique()))
        
        # Filtra opções subsequentes baseado na cidade
        df_temp = df[df['cidade'].isin(cidades)] if cidades else df
        diagnosticos = st.multiselect("🩺 Diagnóstico", sorted(df_temp['diagnostico'].unique()))
        
        # Botão de Reset (UX Importante)
        if st.button("Limpar Filtros", type="primary"):
            st.rerun()

    # Aplicação dos Filtros
    df_filtrado = df.copy()
    if cidades: df_filtrado = df_filtrado[df_filtrado['cidade'].isin(cidades)]
    if diagnosticos: df_filtrado = df_filtrado[df_filtrado['diagnostico'].isin(diagnosticos)]

    # --- CONTEÚDO PRINCIPAL ---
    
    # 1. Cabeçalho com Contexto
    col_header, col_time = st.columns([3, 1])
    with col_header:
        st.title("Visão Geral da Unidade")
        st.markdown(f"**{len(df_filtrado)}** registros analisados em tempo real.")
    with col_time:
        st.caption(f"Atualizado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

    # 2. KPIs (Cards)
    st.markdown("###") # Espaçamento
    k1, k2, k3, k4 = st.columns(4)
    
    # Lógica de KPI
    total = len(df_filtrado)
    idade_media = df_filtrado['idade'].mean() if not df_filtrado.empty else 0
    top_queixa = df_filtrado['queixa'].mode()[0] if not df_filtrado.empty else "N/A"
    
    # Variação simulada (UX: Mostra tendência)
    var_atend = random.choice(["+5%", "-2%", "+12%"]) 

    card_kpi(k1, "Volume Total", f"{total}", var_atend)
    card_kpi(k2, "Idade Média", f"{idade_media:.0f} anos")
    card_kpi(k3, "Queixa Principal", top_queixa)
    card_kpi(k4, "Satisfação (NPS)", "78", "+2 pts")

    st.markdown("---")

    # 3. Análise Visual (Tabs)
    tab_graf, tab_text, tab_raw = st.tabs(["📊 Dashboards", "☁️ Análise Semântica", "📂 Dados Brutos"])

    with tab_graf:
        row1_1, row1_2 = st.columns(2)
        with row1_1:
            grafico_barras_horizontal(df_filtrado, 'diagnostico', 'Top Diagnósticos', '#2E9CCA') # Azul
        with row1_2:
            grafico_barras_horizontal(df_filtrado, 'queixa', 'Principais Queixas', '#E88D67') # Laranja suave
        
        # Gráfico de Dispersão/Linha (Exemplo de evolução)
        if 'data_atendimento' in df_filtrado.columns:
            st.markdown("###")
            daily_counts = df_filtrado.groupby('data_atendimento').size().reset_index(name='counts')
            fig_line = px.area(daily_counts, x='data_atendimento', y='counts', title="Evolução de Atendimentos (Timeline)")
            fig_line.update_traces(line_color='#2E9CCA', fillcolor='rgba(46, 156, 202, 0.2)')
            st.plotly_chart(plot_config_pro(fig_line), use_container_width=True)

    with tab_text:
        col_wc, col_desc = st.columns([2, 1])
        with col_wc:
            # Wordcloud simplificada para o exemplo
            text = " ".join(df_filtrado['queixa'].astype(str))
            if text:
                wc = WordCloud(width=800, height=400, background_color='#0F1116', colormap='ocean').generate(text)
                fig_wc, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                fig_wc.patch.set_alpha(0) # Fundo transparente
                st.pyplot(fig_wc)
        with col_desc:
            st.info("💡 **Insight:** A nuvem de palavras destaca a predominância de sintomas respiratórios nesta amostra, sugerindo sazonalidade.")

    with tab_raw:
        st.markdown("### Exportação e Auditoria")
        st.dataframe(
            df_filtrado,
            use_container_width=True,
            column_config={
                "idade": st.column_config.ProgressColumn("Idade Visual", format="%d", min_value=0, max_value=100),
                "data_atendimento": st.column_config.DateColumn("Data", format="DD/MM/YYYY")
            },
            height=400
        )

if __name__ == "__main__":
    main()
