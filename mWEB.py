import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from processamento import balanceProcessing
from processamento import jumpProcessing
from processamento import tugProcessing
from processamento import ytestProcessing
from processamento import jointSenseProcessing
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.ndimage import uniform_filter1d
from textwrap import dedent

# --------- Config da página ---------
st.set_page_config(
    page_title="Momentum Web",
    page_icon="⚡",
    layout="wide"
)

# ============== I18N: Portão de idioma (NÃO ESCRITO) ==============
# Dicionário básico só para rótulos principais; você pode expandir depois
T = {
    "pt": {
        "choose": "Selecione o idioma",
        "pages": {
            "home": "🏠 Página Inicial",
            "import": "⬆️ Importar Dados",
            "plot": "📈 Visualização Gráfica",
            "export": "📤 Exportar Resultados",
            "refs": "📖 Referências bibliográficas",
        }
    },
    "en": {
        "choose": "Choose your language",
        "pages": {
            "home": "🏠 Home",
            "import": "⬆️ Import Data",
            "plot": "📈 Plots",
            "export": "📤 Export Results",
            "refs": "📖 References",
        }
    },
    "es": {
        "choose": "Elija su idioma",
        "pages": {
            "home": "🏠 Inicio",
            "import": "⬆️ Importar Datos",
            "plot": "📈 Visualización Gráfica",
            "export": "📤 Exportar Resultados",
            "refs": "📖 Referencias bibliográficas",
        }
    },
}

def _set_lang(lang_code: str):
    st.session_state["lang"] = lang_code
    st.query_params["lang"] = lang_code
    st.rerun()

# Lê idioma de URL ou sessão
if "lang" not in st.session_state:
    qp = st.query_params.get("lang", None)
    st.session_state["lang"] = qp if qp in T else None

# Mostra seletor (apenas bandeiras) antes do conteúdo
if st.session_state["lang"] not in T:
    st.markdown("""
        <style>
        .lang-gate {
            position: fixed; inset: 0; display:flex; align-items:center; justify-content:center;
            background: linear-gradient(135deg, #ffffff 0%, #f2f2f2 40%, #e6e6e6 100%);
            z-index: 9999;
        }
        .lang-card {
            background: rgba(255,255,255,.8); padding: 24px; border-radius: 16px;
            box-shadow: 0 8px 30px rgba(0,0,0,.1); text-align:center; min-width: 280px;
        }
        .lang-title { font-size: 1.1rem; margin-bottom: .75rem; opacity: .85; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="lang-gate"><div class="lang-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="lang-title">{T["pt"]["choose"]}</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🇧🇷", type="primary", use_container_width=True, help="Português"):
            _set_lang("pt")
    with c2:
        if st.button("🇺🇸", type="primary", use_container_width=True, help="English"):
            _set_lang("en")
    with c3:
        if st.button("🇪🇸", type="primary", use_container_width=True, help="Español"):
            _set_lang("es")

    st.markdown('</div></div>', unsafe_allow_html=True)
    st.stop()

# Idioma atual
LANG = st.session_state["lang"]
PAGES_LABELS = T[LANG]["pages"]
PAGE_IDS = ["home", "import", "plot", "export", "refs"]
PAGE_LABEL_LIST = [PAGES_LABELS[k] for k in PAGE_IDS]

# ===================== Estilo de fundo (seu CSS) ===================
st.markdown("""
<style>
/* Fundo estilo "alumínio" */
.stApp {
  background: linear-gradient(135deg, #ffffff 0%, #f2f2f2 40%, #e6e6e6 100%);
}

/* Barra superior */
header[data-testid="stHeader"] {
  background: linear-gradient(135deg, #ffffff 0%, #f2f2f2 40%, #e6e6e6 100%) !important;
}

/* Deixa centro e sidebar transparentes para o gradiente aparecer */
.block-container { background: transparent; }
section[data-testid="stSidebar"] { background: transparent; }
</style>
""", unsafe_allow_html=True)

# ====================== Título (mantido) ===========================
st.markdown(
    """
    <h1 style='text-align: center; color: #1E90FF;'>
        Momentum Web
    </h1>
    """,
    unsafe_allow_html=True
)

# =================== Função de carga (mantida) =====================
@st.cache_data
def carregar_dados_generico(arquivo):
    try:
        df = pd.read_csv(arquivo, sep=None, engine='python')
        if df.shape[1] == 5:
            dados = df.iloc[:, 1:5]  # Usa colunas 2 a 5
        elif df.shape[1] == 4:
            dados = df.iloc[:, 0:4]  # Usa todas
        else:
            st.error("O arquivo deve conter 4 ou 5 colunas com cabeçalhos.")
            return None
        dados.columns = ["Tempo", "X", "Y", "Z"]
        return dados
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        return None

# =================== Menu lateral com i18n =========================
selected_label = st.sidebar.radio("📂 Navegue pelas páginas", PAGE_LABEL_LIST)
pagina = PAGE_IDS[PAGE_LABEL_LIST.index(selected_label)]  #

#== Página Inicial ===
if pagina == "home":
    html = dedent("""
        <div style="text-align: justify; font-size: 1.1rem; line-height: 1.6; color: #333333;
            max-width: 900px; margin: auto; background-color: rgba(255,200,255,0.6);
            padding: 20px; border-radius: 8px;">
            <p><b>Bem-vindo ao Momentum Web</b>, a aplicação Web para análise de dados de protocolos de avaliação do
            <i>Momentum Sensors</i>.</p>
            <p>Os protocolos de análise dos dados são baseados em métodos usados em artigos científicos do grupo 
            idealizador do Projeto Momentum compostos por pesquisadores da Universidade Federal do Pará, Universidade do Estado do Pará e Instituto Federal de São Paulo. 
            O projeto representa uma iniciativa de <b>desenvolvimento científico e tecnológico</b> com o objetivo de propor métodos confiáveis de 
            avaliação sensório-motora usando sensores presentes em smartphones.</p>
            <p>Alguns protocolos estarão em desenvolvimento e serão indicados quando for o caso.</p>
            Utilize o <b>menu lateral</b> para navegar entre as diferentes etapas da análise.</p>
        </div>
    """)
    st.markdown(html, unsafe_allow_html=True)

# === Página de Importação ===
elif pagina == "import":
    st.title("⬆️ Importar Dados")

    col1, col2, col3 = st.columns([1, 0.2, 1])
    with col1:
        tipo_teste = st.selectbox(
            "Qual teste você deseja analisar?",
            ["Selecione...", "Equilíbrio", "Salto", "TUG", "Propriocepção", "Y test"]
        )

        if tipo_teste != "Selecione...":
            st.session_state["tipo_teste"] = tipo_teste

        if tipo_teste == "Equilíbrio":
            st.subheader("🧍🏽‍♀️ Importar dados de Equilíbrio")
            arquivo = st.file_uploader(
                "Selecione o arquivo de equilíbrio (CSV ou TXT)", type=["csv", "txt"])
            if arquivo is not None:
                dados = carregar_dados_generico(arquivo)
                if dados is not None:
                    st.success('Dados carregados com sucesso')
                    st.session_state["dados"] = dados

        elif tipo_teste == "Salto":
            st.subheader("🤸 Importar dados de Salto")
            arquivo = st.file_uploader(
                "Selecione o arquivo de salto (CSV ou TXT)", type=["csv", "txt"])
            if arquivo is not None:
                dados = carregar_dados_generico(arquivo)
                if dados is not None:
                    st.success('Dados carregados com sucesso')
                    st.session_state["dados"] = dados

        elif tipo_teste == "TUG":
            st.subheader("📱 Importar dados dos sensores")
            arquivo = st.file_uploader(
                "Selecione o arquivo do acelerômetro (CSV ou TXT)", type=["csv", "txt"])
            if arquivo is not None:
                dados_acc = carregar_dados_generico(arquivo)
                if dados_acc is not None:
                    st.success("Arquivo carregado com sucesso!")
                    st.dataframe(dados_acc.head())
                    st.session_state["dados_acc"] = dados_acc
                    st.session_state["dados"] = dados_acc
                    arquivo = st.file_uploader("Selecione o arquivo do giroscópico (CSV ou TXT)", type=["csv", "txt"])
                    if arquivo is not None:
                        dados_gyro = carregar_dados_generico(arquivo)
                        if dados_gyro is not None:
                            st.success("Arquivo carregado com sucesso!")
                            st.dataframe(dados_gyro.head())
                            st.session_state["dados_gyro"] = dados_gyro
        
        elif tipo_teste == "Y test":
            st.subheader("📱 Importar dados dos sensores")
            arquivo = st.file_uploader(
                "Selecione o arquivo da coluna vertebral (CSV ou TXT)", type=["csv", "txt"])
            if arquivo is not None:
                dados_acc_coluna = carregar_dados_generico(arquivo)
                if dados_acc_coluna is not None:
                    st.success("Arquivo carregado com sucesso!")
                    st.dataframe(dados_acc_coluna.head())
                    st.session_state["dados_acc_coluna"] = dados_acc_coluna
                    st.session_state["dados"] = dados_acc_coluna
                    arquivo_2 = st.file_uploader("Selecione o arquivo do joelho (CSV ou TXT)", type=["csv", "txt"])
                    if arquivo_2 is not None:
                        dados_acc_joelho = carregar_dados_generico(arquivo_2)
                        if dados_acc_joelho is not None:
                            st.success("Arquivo carregado com sucesso!")
                            st.dataframe(dados_acc_joelho.head())
                            st.session_state["dados_acc_joelho"] = dados_acc_joelho

        elif tipo_teste == "Propriocepção":
            st.subheader("📦 Importar dados de Propriocepção")
            arquivo = st.file_uploader(
                "Selecione o arquivo de propriocepção (CSV ou TXT)", type=["csv", "txt"])
            if arquivo is not None:
                dados = carregar_dados_generico(arquivo)
                if dados is not None:
                    st.success('Dados carregados com sucesso')
                    st.session_state["dados"] = dados
        elif tipo_teste == "Selecione...":
            st.info("Selecione um tipo de teste para continuar.")

    with col3:
        if tipo_teste == "Equilíbrio":
            st.markdown(
            """
            <h1 style='text-align: center; color: #1E90FF;'>
            🧍🏽‍♀️Equilíbrio estático
            </h1>
            """,
            unsafe_allow_html=True
            )
            html = dedent("""
            <div style="text-align: justify; font-size: 1.1rem; line-height: 1.6; color: #333333;
            max-width: 900px; margin: auto; background-color: rgba(255,200,255,0.6);
            padding: 20px; border-radius: 8px;">
    
            <p>
            A avaliação do equilíbrio estático usando o Momentum Sensors foi baseada nos artigos de 
            <a href="https://www.mdpi.com/2075-4426/12/7/1019" target="_blank" style="color:#1E90FF; text-decoration:none;">Rodrigues et al. (2022)</a>, 
            <a href="https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2023.1277408/full" target="_blank" style="color:#1E90FF; text-decoration:none;">Correa et al. (2023)</a> 
            e 
            <a href="https://www.scielo.br/j/aabc/a/7z5HDVZKYVMxfWm8HxcJqZG/?lang=en&format=pdf" target="_blank" style="color:#1E90FF; text-decoration:none;">Fernandes et al. (2024)</a>.
            </p>

            <p>
            É necessário fixar o smartphone na coluna lombar do paciente e pedir para que ele não se movimente ou fale durante o tempo de registro.
            </p>

            </div>
            """)
            st.markdown(html, unsafe_allow_html=True)
        elif tipo_teste == 'Salto':
            st.markdown(
            """
            <h1 style='text-align: center; color: #1E90FF;'>
            🤸Salto vertical
            </h1>
            """,
            unsafe_allow_html=True
            )
            html = dedent("""
            <div style="text-align: justify; font-size: 1.1rem; line-height: 1.6; color: #333333;
            max-width: 900px; margin: auto; background-color: rgba(255,200,255,0.6);
            padding: 20px; border-radius: 8px;">
    
            <p>
            A avaliação do salto vertical usando o Momentum Sensors foi baseada nos artigos de 
            <a href="https://www.mdpi.com/1424-8220/23/13/6022" target="_blank" style="color:#1E90FF; text-decoration:none;">Moreno-Pérez et al. (2023)</a>  
            e <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC5454547/" target="_blank" style="color:#1E90FF; text-decoration:none;">Mateos-Angulo et al. (2015)</a>. 
            </p>

            </div>
            """)
            st.markdown(html, unsafe_allow_html=True)
        elif tipo_teste == "TUG":
            st.markdown(
            """
            <h1 style='text-align: center; color: #1E90FF;'>
            Timed Up and Go instrumentado
            </h1>
            """,
            unsafe_allow_html=True
            )
            html = dedent("""
            <div style="text-align: justify; font-size: 1.1rem; line-height: 1.6; color: #333333;
            max-width: 900px; margin: auto; background-color: rgba(255,200,255,0.6);
            padding: 20px; border-radius: 8px;">
    
            <p>
            A avaliação do Timed Up ang Go instrumentado usando o Momentum Sensors foi baseada nos artigos de 
            <a href="https://www.mdpi.com/1424-8220/24/9/2918" target="_blank" style="color:#1E90FF; text-decoration:none;">Santos et al. (2024)</a>, 
            <a href="https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2023.1277408/full" target="_blank" style="color:#1E90FF; text-decoration:none;">Correa et al. (2023)</a> 
            e 
            <a href="https://www.scielo.br/j/aabc/a/7z5HDVZKYVMxfWm8HxcJqZG/?lang=en&format=pdf" target="_blank" style="color:#1E90FF; text-decoration:none;">Fernandes et al. (2024)</a>.
            </p>

            </div>
            """)
            st.markdown(html, unsafe_allow_html=True)
        elif tipo_teste == "Propriocepção":
            st.markdown(
            """
            <h1 style='text-align: center; color: #1E90FF;'>
            Sensação de posicionamento articular
            </h1>
            """,
            unsafe_allow_html=True
            )
            html = dedent("""
            <div style="text-align: justify; font-size: 1.1rem; line-height: 1.6; color: #333333;
            max-width: 900px; margin: auto; background-color: rgba(255,200,255,0.6);
            padding: 20px; border-radius: 8px;">
    
            <p>
            A avaliação da sensação de posi~cionamento articular usando o Momentum Sensors foi baseada no artigo de 
            <a href="https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1561241/full" style="color:#1E90FF; text-decoration:none;">Almeida et al. (2025)</a>. 
            É preciso mensurar a amplitude articular inicial usando goniômetro para adicionar à variações articulares desta posição inicial. 
            </p>

            </div>
            """)
            st.markdown(html, unsafe_allow_html=True)  
        elif tipo_teste == "Y test":
            st.markdown(
            """
            <h1 style='text-align: center; color: #1E90FF;'>
            Y test
            </h1>
            """,
            unsafe_allow_html=True
            )
            html = dedent("""
            <div style="text-align: justify; font-size: 1.1rem; line-height: 1.6; color: #333333;
            max-width: 900px; margin: auto; background-color: rgba(255,200,255,0.6);
            padding: 20px; border-radius: 8px;">
    
            <p>
            A avaliação do equilíbrio dinâmico pelo Y test está em desenvolvimento sob coordenação do Prof. Dr. André dos Santos Cabral da Universidade do Estado do Pará. 
            </p>

            </div>
            """)
            st.markdown(html, unsafe_allow_html=True)     
        else:
            st.title('Men at working')

    









