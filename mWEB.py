import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar  # <-- corrigido
from processamento import balanceProcessing, jumpProcessing, tugProcessing, ytestProcessing, jointSenseProcessing
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.ndimage import uniform_filter1d
from textwrap import dedent

# --------- Config da página ---------
st.set_page_config(page_title="Momentum Web", page_icon="⚡", layout="wide")

# --------- Estilo ---------
st.markdown(
    """
    <style>
      .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #f2f2f2 40%, #e6e6e6 100%);
      }
      header[data-testid="stHeader"] {
        background: linear-gradient(135deg, #ffffff 0%, #f2f2f2 40%, #e6e6e6 100%) !important;
      }
      .block-container { background: transparent; }
      section[data-testid="stSidebar"] { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center; color: #1E90FF;'>Momentum Web</h1>", unsafe_allow_html=True)

@st.cache_data
def carregar_dados_generico(arquivo):
    """
    Lê CSV com 4 ou 5 colunas.
    - 5 colunas: descarta a 1ª (ex.: índice/metadata) e usa as colunas 2..5
    - 4 colunas: usa todas
    Retorna DataFrame com colunas: ["Tempo", "X", "Y", "Z"] ou None em caso de erro.
    """
    try:
        df = pd.read_csv(arquivo, sep=None, engine='python')  # autodetecta separador

        if df.shape[1] == 5:
            dados = df.iloc[:, 1:5].copy()  # usa colunas 2..5
        elif df.shape[1] == 4:
            dados = df.iloc[:, 0:4].copy()  # usa todas
        else:
            st.error("O arquivo deve conter 4 ou 5 colunas com cabeçalhos.")
            return None

        dados.columns = ["Tempo", "X", "Y", "Z"]
        return dados

    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        return None

pagina = st.sidebar.radio("📂 Navegue pelas páginas", [ "🏠 Página Inicial", "⬆️ Importar Dados", "📈 Visualização Gráfica", "📤 Exportar Resultados", "📖 Referências bibliográficas" ])

# === Página Inicial ===
if pagina == "🏠 Página Inicial": # texto descritivo mais bonito
    html = dedent(""" <div style="text-align: justify; font-size: 1.1rem; line-height: 1.6; color: #333333; max-width: 900px; margin: auto; background-color: rgba(255,200,255,0.6); padding: 20px; border-radius: 8px;">
    <p><b>Bem-vindo ao Momentum Web</b>,
    a aplicação Web para análise de dados de protocolos de avaliação do <i>Momentum Sensors</i>.
    </p> 
    <p>Os protocolos de análise dos dados são baseados em métodos usados em artigos científicos do grupo idealizador do Projeto Momentum compostos por pesquisadores da Universidade Federal do Pará, Universidade do Estado do Pará e Instituto Federal de São Paulo. O projeto representa uma iniciativa de <b>desenvolvimento científico e tecnológico</b> com o objetivo de propor métodos confiáveis de avaliação sensório-motora usando sensores presentes em smartphones.</p>
    <p>Alguns protocolos estarão em desenvolvimento e serão indicados quando for o caso.</p>
    Utilize o <b>menu lateral</b> para navegar entre as diferentes etapas da análise.</p>
    </div> """) 

    st.markdown(html, unsafe_allow_html=True)

# === Página de Importação ===
elif pagina == "⬆️ Importar Dados":
    st.title("⬆️ Importar Dados")
    col1, col2, col3 = st.columns([1, 0.2, 1])
    with col1:
        tipo_teste = st.selectbox( "Qual teste você deseja analisar?", ["Selecione...", "Equilíbrio", "Salto", "TUG", "Propriocepção", "Y test"] )
        if tipo_teste != "Selecione...":
            st.session_state["tipo_teste"] = tipo_teste
            if tipo_teste == "Equilíbrio":
                st.subheader("🧍🏽‍♀️ Importar dados de Equilíbrio")
                arquivo = st.file_uploader( "Selecione o arquivo de equilíbrio (CSV ou TXT)", type=["csv", "txt"])
                if arquivo is not None:
                    dados = carregar_dados_generico(arquivo)
                    if dados is not None:
                        st.success('Dados carregados com sucesso')
                        st.session_state["dados"] = dados
                        st.session_state["tipo_teste"] = tipo_teste
            elif tipo_teste == "Salto":
                 st.subheader("🤸 Importar dados de Salto")
                 arquivo = st.file_uploader( "Selecione o arquivo de salto (CSV ou TXT)", type=["csv", "txt"])
                 if arquivo is not None:
                     dados = carregar_dados_generico(arquivo)
                     if dados is not None:
                         st.success('Dados carregados com sucesso')
                         st.session_state["dados"] = dados
                         st.session_state["tipo_teste"] = tipo_teste
            elif tipo_teste == "TUG":
                 st.subheader("📱 Importar dados dos sensores")
                 arquivo = st.file_uploader( "Selecione o arquivo do acelerômetro (CSV ou TXT)", type=["csv", "txt"])
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
                                 st.session_state["tipo_teste"] = tipo_teste
            elif tipo_teste == "Y test":
                 st.subheader("📱 Importar dados dos sensores")
                 arquivo = st.file_uploader( "Selecione o arquivo da coluna vertebral (CSV ou TXT)", type=["csv", "txt"])
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
                                 st.session_state["tipo_teste"] = tipo_teste
            elif tipo_teste == "Propriocepção":
                 st.subheader("📦 Importar dados de Propriocepção")
                 arquivo = st.file_uploader( "Selecione o arquivo de propriocepção (CSV ou TXT)", type=["csv", "txt"])
                 if arquivo is not None:
                     dados = carregar_dados_generico(arquivo)
                     if dados is not None:
                         st.success('Dados carregados com sucesso')
                         st.session_state["dados"] = dados
                         st.session_state["tipo_teste"] = tipo_teste
        elif tipo_teste == "Selecione...":
             st.info("Selecione um tipo de teste para continuar.")
    with col3:
        if tipo_teste == "Equilíbrio":
            st.markdown( """ <h1 style='text-align: center; color: #1E90FF;'> 🧍🏽‍♀️Equilíbrio estático </h1> """, unsafe_allow_html=True )
            html = dedent(""" <div style="text-align: justify; font-size: 1.1rem; line-height: 1.6; color: #333333; max-width: 900px; margin: auto; background-color: rgba(255,200,255,0.6); padding: 20px; border-radius: 8px;">
            <p> A avaliação do equilíbrio estático usando o Momentum Sensors foi baseada nos artigos de 
            <a href="https://www.mdpi.com/2075-4426/12/7/1019" target="_blank" style="color:#1E90FF; text-decoration:none;">Rodrigues et al. (2022)</a>, 
            <a href="https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2023.1277408/full" target="_blank" style="color:#1E90FF; text-decoration:none;">Correa et al. (2023)</a> e 
            <a href="https://www.scielo.br/j/aabc/a/7z5HDVZKYVMxfWm8HxcJqZG/?lang=en&format=pdf" target="_blank" style="color:#1E90FF; text-decoration:none;">Fernandes et al. (2024)</a>. </p> 
            <p> É necessário fixar o smartphone na coluna lombar do paciente e pedir para que ele não se movimente ou fale durante o tempo de registro. </p> </div> """)
            st.markdown(html, unsafe_allow_html=True)
        
        elif tipo_teste == 'Salto':
            st.markdown( """ <h1 style='text-align: center; color: #1E90FF;'> 🤸Salto vertical </h1> """, unsafe_allow_html=True )
            html = dedent(""" <div style="text-align: justify; font-size: 1.1rem; line-height: 1.6; color: #333333; max-width: 900px; margin: auto; background-color: rgba(255,200,255,0.6); padding: 20px; border-radius: 8px;"> <p> A avaliação do salto vertical usando o Momentum Sensors foi baseada nos artigos de
            <a href="https://www.mdpi.com/1424-8220/23/13/6022" target="_blank" style="color:#1E90FF; text-decoration:none;">Moreno-Pérez et al. (2023)</a> e 
            <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC5454547/" target="_blank" style="color:#1E90FF; text-decoration:none;">Mateos-Angulo et al. (2015)</a>. </p> </div> """)
            st.markdown(html, unsafe_allow_html=True)
        
        elif tipo_teste == "TUG":
            st.markdown( """ <h1 style='text-align: center; color: #1E90FF;'> Timed Up and Go instrumentado </h1> """, unsafe_allow_html=True )
            html = dedent(""" <div style="text-align: justify; font-size: 1.1rem; line-height: 1.6; color: #333333; max-width: 900px; margin: auto; background-color: rgba(255,200,255,0.6); padding: 20px; border-radius: 8px;"> <p> A avaliação do Timed Up ang Go instrumentado usando o Momentum Sensors foi baseada nos artigos de
            <a href="https://www.mdpi.com/1424-8220/24/9/2918" target="_blank" style="color:#1E90FF; text-decoration:none;">Santos et al. (2024)</a>,
            <a href="https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2023.1277408/full" target="_blank" style="color:#1E90FF; text-decoration:none;">Correa et al. (2023)</a> e
            <a href="https://www.scielo.br/j/aabc/a/7z5HDVZKYVMxfWm8HxcJqZG/?lang=en&format=pdf" target="_blank" style="color:#1E90FF; text-decoration:none;">Fernandes et al. (2024)</a>. 
            O protocolo de análise está sob expansão no momento e em breve terá mais características extraídas.</p> </div> """)
            st.markdown(html, unsafe_allow_html=True)
        
        elif tipo_teste == "Propriocepção":
            st.markdown( """ <h1 style='text-align: center; color: #1E90FF;'> Sensação de posicionamento articular </h1> """, unsafe_allow_html=True )
            html = dedent(""" <div style="text-align: justify; font-size: 1.1rem; line-height: 1.6; color: #333333; max-width: 900px; margin: auto; background-color: rgba(255,200,255,0.6); padding: 20px; border-radius: 8px;"> <p> A avaliação da sensação de posi~cionamento articular usando o Momentum Sensors foi baseada no artigo de
            <a href="https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1561241/full" style="color:#1E90FF; text-decoration:none;">Almeida et al. (2025)</a>.
            É preciso mensurar a amplitude articular inicial usando goniômetro para adicionar à variações articulares desta posição inicial. </p> </div> """)
            st.markdown(html, unsafe_allow_html=True)
        
        elif tipo_teste == "Y test":
            st.markdown( """ <h1 style='text-align: center; color: #1E90FF;'> Y test </h1> """, unsafe_allow_html=True )
            html = dedent(""" <div style="text-align: justify; font-size: 1.1rem; line-height: 1.6; color: #333333; max-width: 900px; margin: auto; background-color: rgba(255,200,255,0.6); padding: 20px; border-radius: 8px;"> <p> A avaliação do equilíbrio dinâmico pelo Y test está em desenvolvimento sob coordenação do Prof. Dr. André dos Santos Cabral da Universidade do Estado do Pará. </p> </div> """)
            st.markdown(html, unsafe_allow_html=True)
        
        else: st.title('Men at working')
# === Página de Visualização Gráfica ===
elif pagina == "📈 Visualização Gráfica":
    st.title("📈 Visualização Gráfica")
    if "dados" in st.session_state and "tipo_teste" in st.session_state:
        tipo_teste = st.session_state["tipo_teste"]
        st.subheader(f"📊 Visualização - {tipo_teste}") 
        if tipo_teste == "Equilíbrio":
            dados = st.session_state["dados"]
            tempo, ml, ap, freqs, psd_ml, psd_ap = balanceProcessing.processar_equilibrio( dados, 0, 0, 0, 0, 8)
            max_val = len(tempo)
            col1, col2, col3 = st.columns(3)
            with col1:
                startRec = st.number_input( 'Indique o início do registro', value=0, step=1, max_value=max_val)
            with col2:
                endRec = st.number_input( 'Indique o final do registro', value=max_val, step=1, max_value=max_val)
            with col3:
                filter = st.number_input( 'Indique o filtro passa-baixa', value=8.0, step=0.1, max_value=40.0)
            st.session_state["intervalo"] = startRec, endRec, filter
            showRec = st.checkbox('Mostrar o dado original', value=True)
            tempo, ml, ap, freqs, psd_ml, psd_ap = balanceProcessing.processar_equilibrio( dados, 0, 0, 0, 0, 49)
            tempo_sel, ml_sel, ap_sel, freqs_sel, psd_ml_sel, psd_ap_sel = balanceProcessing.processar_equilibrio( dados, startRec, endRec, 1, 0, filter)
            if startRec > endRec:
                st.error( 'Valor do início do registro não pode ser maior que o do final do registro')
            else:
                if endRec > max_val:
                    st.error( 'Valor do início do registro não pode ser maior que o do final do registro')
                else:
                    min_ml = np.min(ml)
                    max_ml = np.max(ml)
                    min_ap = np.min(ap)
                    max_ap = np.max(ap)
                    limite = max(np.abs(min_ml), np.abs( max_ml), np.abs(min_ap), np.abs(max_ap))
                    if limite < 0.5:
                        limite = 0.5
                    elif limite >= 0.5 and limite < 5:
                        limite = 5
                    elif limite >= 5 and limite < 10:
                        limite = 10
                    else:
                        limite = 50
                    limite = 1.5
                    # Cria figura com GridSpec personalizado
                    fig = plt.figure(figsize=(8, 10))
                    gs = gridspec.GridSpec( 5, 4, figure=fig, hspace=0.8, wspace=0.6) # Gráfico 1: ocupa 2x2 blocos (esquerda acima)
                    rms_ml, rms_ap, total_deviation, ellipse_area, avg_x, avg_y, width, height, angle, direction = balanceProcessing.processar_equilibrio( dados, startRec, endRec, 1, 1, filter)
                    ellipse = Ellipse(xy=(avg_x, avg_y), width=width, height=height, angle=angle, alpha=0.5, color='blue', zorder=10)
                    ax1 = fig.add_subplot(gs[0:2, 0:2])
                    if showRec:
                        ax1.plot(ml, ap, color='tomato', linewidth=0.5)
                    ax1.plot( ml_sel[startRec:endRec], ap_sel[startRec:endRec], color='black', linewidth=0.8)
                    ax1.set_xlabel(r'Aceleração ML (m/s$^2$)', fontsize=8)
                    ax1.set_ylabel(r'Aceleração AP (m/s$^2$)', fontsize=8)
                    ax1.set_xlim(-limite, limite)
                    ax1.set_ylim(-limite, limite)
                    ax1.tick_params(axis='both', labelsize=8)
                    ax1.add_patch(ellipse) # Gráfico 2: ocupa linha superior direita (metade superior)
                                        
                    ax2 = fig.add_subplot(gs[0, 2:])
                    if showRec:
                        ax2.plot(tempo, ml, color='tomato', linewidth=0.5)
                    ax2.plot( tempo_sel[startRec:endRec], ml_sel[startRec:endRec], color='black', linewidth=0.8)
                    ax2.set_xlabel('Tempo (s)', fontsize=8)
                    ax2.set_ylabel(r'Aceleração ML (m/s$^2$)', fontsize=8)
                    ax2.set_xlim(0, max(tempo))
                    ax2.set_ylim(-limite, limite)
                    ax2.tick_params(axis='both', labelsize=8) # Gráfico 3: linha do meio à direita
                                        
                    ax3 = fig.add_subplot(gs[1, 2:])
                    if showRec:
                        ax3.plot(tempo, ap, color='tomato', linewidth=0.5)
                    ax3.plot( tempo_sel[startRec:endRec], ap_sel[startRec:endRec], color='black', linewidth=0.8)
                    ax3.set_xlabel('Tempo (s)', fontsize=8)
                    ax3.set_ylabel(r'Aceleração AP (m/s$^2$)', fontsize=8)
                    ax3.set_xlim(0, max(tempo))
                    ax3.set_ylim(-limite, limite)
                    ax3.tick_params(axis='both', labelsize=8) # Gráfico 4: canto inferior esquerdo
                                        
                    ax4 = fig.add_subplot(gs[2:4, 0:2])
                    if showRec:
                        ax4.plot(freqs, psd_ml, color='tomato', linewidth=0.5)
                    ax4.plot(freqs_sel, psd_ml_sel, 'k')
                    ax4.set_xlim(-0.1, 8)
                    ax4.set_ylim(0, limite*0.025)
                    ax4.set_xlabel('Frequência temporal (Hz)', fontsize=8)
                    ax4.set_ylabel(r'Aceleração ML (m/s$^2$)', fontsize=8)
                    ax4.tick_params(axis='both', labelsize=8) # Gráfico 5: canto inferior direito
                                        
                    ax5 = fig.add_subplot(gs[2:4, 2:])
                    if showRec:
                        ax5.plot(freqs, psd_ap, color='tomato', linewidth=0.5)
                    ax5.plot(freqs_sel, psd_ap_sel, 'k')
                    ax5.set_xlim(-0.1, 8)
                    ax5.set_ylim(0, limite*0.025)
                    ax5.set_xlabel('Frequência temporal (Hz)', fontsize=8)
                    ax5.set_ylabel(r'Aceleração AP (m/s$^2$)', fontsize=8)
                    ax5.tick_params(axis='both', labelsize=8) # Exibe no Streamlit st.pyplot(fig)
                    st.pyplot(fig)
        if tipo_teste == "Salto":
            col1, col2, col3 = st.columns([0.4, 1, 0.4])
            dados = st.session_state["dados"]
            tempo, salto, startJump, endJump, altura, tempo_voo, m1, m2, veloc, desloc, istart, iend = jumpProcessing.processar_salto( dados, 8)
            with col2:
                fig, ax = plt.subplots()
                ax.plot(tempo[istart-100:iend+100], salto[istart - 100:iend+100], linewidth=0.8, color='black')
                ax.axvline(startJump, color='green', linestyle='--', label='Início Voo', linewidth=0.8)
                ax.axvline(endJump, color='red', linestyle='--', label='Fim Voo', linewidth=0.8)
                ax.set_xlabel('Tempo (s)')
                ax.set_ylabel('Aceleração vertical (m/s²)')
                ax.legend() 
                st.pyplot(fig)
        if tipo_teste == "TUG":
            dados_acc = st.session_state["dados_acc"]
            dados_gyro = st.session_state["dados_gyro"]
            baseline_onset = st.number_input('Indique o momento inicial da baseline do início do teste (s)', value=0.0)
            baseline_offset = st.number_input('Indique o momento inicial da baseline do final do teste (s)', value=(np.max(dados_acc["Tempo"])/1000)-2.5)
            st.session_state["baseline_onset"] = baseline_onset
            st.session_state["baseline_offset"] = baseline_offset
            col1, col2 = st.columns(2)
            
            t_novo_acc, v_acc, ml_acc, z_acc_filtrado, norma_acc_filtrado, t_novo_gyro, v_gyro, ml_gyro, z_gyro_filtrado, norma_gyro_filtrado,start_test,stop_test,idx,idx_ml,idx_acc_ap,idx_acc_v,duration = tugProcessing.processar_tug(dados_acc,dados_gyro,2,1.25,baseline_onset,baseline_offset)
            vertical_squared = np.sqrt(v_gyro**2)
            lat1 = idx[1][0]
            lat2 = idx[1][1]
            amp1 = vertical_squared[idx[0][0]]
            amp2 = vertical_squared[idx[0][1]]
            if lat1 > lat2:
                G1_lat = lat2
                G1_amp = amp2
                G2_lat = lat1
                G2_amp = amp1
            else:
                G1_lat = lat1
                G1_amp = amp1
                G2_lat = lat2
                G2_amp = amp2
            ml_squared = np.sqrt(ml_gyro**2)
            lat1 = idx_ml[1][0]
            lat2 = idx_ml[1][1]
            amp1 = ml_squared[idx_ml[0][0]]
            amp2 = ml_squared[idx_ml[0][1]]
            if lat1 > lat2:
                G0_lat = lat2
                G0_amp = amp2
                G4_lat = lat1
                G4_amp = amp1
            else:
                G0_lat = lat1
                G0_amp = amp1
                G4_lat = lat2
                G4_amp = amp2
            acc_ap_squared = np.sqrt(z_acc_filtrado**2)
            lat1 = idx_acc_ap[1][0]
            lat2 = idx_acc_ap[1][1]
            amp1 = acc_ap_squared[idx_acc_ap[0][0]]
            amp2 = acc_ap_squared[idx_acc_ap[0][1]]
            if lat1 > lat2:
                A1_lat = lat2
                A1_amp = amp2
                A2_lat = lat1
                A2_amp = amp1
            else:
                A1_lat = lat1
                A1_amp = amp1
                A2_lat = lat2
                A2_amp = amp2
            acc_v_squared = np.sqrt(v_acc**2)
            lat1 = idx_acc_v[1][0]
            lat2 = idx_acc_v[1][1]
            amp1 = acc_v_squared[idx_acc_v[0][0]]
            amp2 = acc_v_squared[idx_acc_v[0][1]]
            if lat1 > lat2:
                A1v_lat = lat2
                A1v_amp = amp2
                A2v_lat = lat1
                A2v_amp = amp1
            else:
                A1v_lat = lat1
                A1v_amp = amp1
                A2v_lat = lat2
                A2v_amp = amp2
            with col1:
                fig1, ax1 = plt.subplots()
                ax1.plot(t_novo_acc, norma_acc_filtrado, linewidth=0.8, color='black')
                #ax1.axvline(start_test, color='green', linestyle='--', label='Início', linewidth=0.8)
                #ax1.axvline(stop_test, color='red', linestyle='--', label='Final', linewidth=0.8)
                ax1.set_xlim(start_test-5,stop_test+5)
                ax1.set_xlabel('Tempo (s)')
                ax1.set_ylabel('Aceleração norma (m/s²)')
                ax1.legend()
                st.pyplot(fig1)
                fig2, ax2 = plt.subplots()
                ax2.plot(t_novo_acc, np.sqrt(ml_acc**2), linewidth=0.8, color='black')
                #ax2.axvline(start_test, color='green', linestyle='--', label='Início', linewidth=0.8)
                #ax2.axvline(stop_test, color='red', linestyle='--', label='Final', linewidth=0.8)
                ax2.set_xlim(start_test-5,stop_test+5)
                ax2.set_xlabel('Tempo (s)')
                ax2.set_ylabel('Aceleração ML (m/s²)')
                ax2.legend()
                st.pyplot(fig2)
                fig3, ax3 = plt.subplots()
                ax3.plot(t_novo_acc, np.sqrt(v_acc**2), linewidth=0.8, color='black')
                ax3.plot(A1v_lat,A1v_amp,'ro')
                ax3.plot(A2v_lat,A2v_amp,'ro')
                #ax3.axvline(start_test, color='green', linestyle='--', label='Início', linewidth=0.8)
                #ax3.axvline(stop_test, color='red', linestyle='--', label='Final', linewidth=0.8)
                ax3.set_xlim(start_test-5,stop_test+5)
                ax3.set_xlabel('Tempo (s)')
                ax3.set_ylabel('Aceleração vertical (m/s²)')
                ax3.legend()
                st.pyplot(fig3)
                fig4, ax4 = plt.subplots()
                ax4.plot(t_novo_acc, np.sqrt(z_acc_filtrado**2), linewidth=0.8, color='black')
                ax4.plot(A1_lat,A1_amp,'ro')
                ax4.plot(A2_lat,A2_amp,'ro')
                #ax4.axvline(start_test, color='green', linestyle='--', label='Início', linewidth=0.8)
                #ax4.axvline(stop_test, color='red', linestyle='--', label='Final', linewidth=0.8)
                ax4.set_xlim(start_test-5,stop_test+5)
                ax4.set_xlabel('Tempo (s)')
                ax4.set_ylabel('Aceleração AP (m/s²)')
                ax4.legend()
                st.pyplot(fig4)
            with col2:
                fig5, ax5 = plt.subplots()
                ax5.plot(t_novo_gyro, norma_gyro_filtrado, linewidth=0.8, color='black')
                #ax5.axvline(start_test, color='green', linestyle='--', label='Início', linewidth=0.8)
                #ax5.axvline(A1v_lat, color='blue', linestyle='--', label='A1 v', linewidth=0.8)
                #ax5.axvline(A1_lat, color='orange', linestyle='--', label='A1 AP', linewidth=0.8)
                #ax5.axvline(G1_lat, color='black', linestyle='--', label='G1', linewidth=0.8)
                #ax5.axvline(G2_lat, color='black', linestyle='--', label='G2', linewidth=0.8)
                #ax5.axvline(G4_lat, color='cyan', linestyle='--', label='G4', linewidth=0.8)
                #ax5.axvline(A2v_lat, color='yellow', linestyle='--', label='A2 v', linewidth=0.8)
                #ax5.axvline(A2_lat, color='gray', linestyle='--', label='A2 AP', linewidth=0.8)
                #ax5.axvline(stop_test, color='red', linestyle='--', label='Final', linewidth=0.8)
                ax5.set_xlim(start_test-5,stop_test+5)
                ax5.set_xlabel('Tempo (s)')
                ax5.set_ylabel('Velocidade angular norma (rad/s)')
                ax5.legend()
                st.pyplot(fig5)
                fig6, ax6 = plt.subplots()
                ax6.plot(t_novo_gyro, np.sqrt(v_gyro**2), linewidth=0.8, color='black')
                ax6.plot(G1_lat,G1_amp,'ro')
                ax6.plot(G2_lat,G2_amp,'ro')
                #ax6.axvline(start_test, color='green', linestyle='--', label='Início', linewidth=0.8)
                #ax6.axvline(stop_test, color='red', linestyle='--', label='Final', linewidth=0.8)
                ax6.set_xlim(start_test-5,stop_test+5)
                ax6.set_xlabel('Tempo (s)')
                ax6.set_ylabel('Velocidade angular Vertical (rad/s)')
                ax6.legend()
                st.pyplot(fig6)
                fig7, ax7 = plt.subplots()
                ax7.plot(t_novo_gyro, np.sqrt(ml_gyro**2), linewidth=0.8, color='black')
                ax7.plot(G0_lat,G0_amp,'ro')
                ax7.plot(G4_lat,G4_amp,'ro')
                #ax7.axvline(start_test, color='green', linestyle='--', label='Início', linewidth=0.8)
                #ax7.axvline(stop_test, color='red', linestyle='--', label='Final', linewidth=0.8)
                ax7.set_xlim(start_test-5,stop_test+5)
                ax7.set_xlabel('Tempo (s)')
                ax7.set_ylabel('Velocidade angular ML (rad/s)')
                ax7.legend()
                st.pyplot(fig7)
                fig8, ax8 = plt.subplots()
                ax8.plot(t_novo_gyro, np.sqrt(z_gyro_filtrado**2), linewidth=0.8, color='black')
                #ax8.axvline(start_test, color='green', linestyle='--', label='Início', linewidth=0.8)
                #ax8.axvline(stop_test, color='red', linestyle='--', label='Final', linewidth=0.8)
                ax8.set_xlim(start_test-5,stop_test+5)
                ax8.set_xlabel('Tempo (s)')
                ax8.set_ylabel('Velocidade angular AP (rad/s)')
                ax8.legend()
                st.pyplot(fig8)
        if tipo_teste == "Y test":
            dados = st.session_state["dados_acc_coluna"]
            dados2 = st.session_state["dados_acc_joelho"]
            tempo, ml, ap, v= ytestProcessing.processar_ytest1(dados,8)
            max_val = 5000
            col1, col2, col3 = st.columns(3)
            with col1:
                startRec = st.number_input( 'Indique o início do registro', value=0, step=1, max_value=max_val)
            with col2:
                endRec = st.number_input( 'Indique o final do registro', value=max_val, step=1, max_value=max_val)
            with col3:
                filter = st.number_input( 'Indique o filtro passa-baixa', value=8.0, step=0.1, max_value=40.0)
            showRec = st.checkbox('Mostrar o dado original', value=True)
            tempo, ml, ap, v = ytestProcessing.processar_ytest1(dados[0:len(dados)-10],filter)
            tempo_2, ml_2, ap_2, v_2= ytestProcessing.processar_ytest2(dados2[0:len(dados2)-10],filter)
            col1, col2 = st.columns(2)
            tempo_sel, ml_sel, ap_sel, v_sel = ytestProcessing.processar_ytest1( dados[startRec:endRec], filter)
            tempo_sel_2, ml_2_sel, ap_2_sel, v_2_sel = ytestProcessing.processar_ytest2( dados2[startRec:endRec], filter)
            picoSaltoCintura = np.max(v[0:1000])
            for index,valor in enumerate(v):
                if valor == picoSaltoCintura:
                    onsetCintura = index 
                    tempo = tempo - tempo[onsetCintura]
                    break
            picoSaltoJoelho = np.max(v_2[0:1000])
            for index,valor in enumerate(v_2):
                if valor == picoSaltoJoelho:
                    onsetJoelho = index
                    tempo_2 = tempo_2 - tempo_2[onsetJoelho]
                    break
            picoSaltoCintura_sel = np.max(v_sel[0:1000])
            for index,valor in enumerate(v_sel):
                if valor == picoSaltoCintura_sel:
                    onsetCintura_sel = index
                    tempo_sel = tempo_sel - tempo_sel[onsetCintura_sel]
                    break
            picoSaltoJoelho_sel = np.max(v_2_sel[0:1000])
            for index,valor in enumerate(v_2_sel):
                if valor == picoSaltoJoelho_sel:
                    onsetJoelho_sel = index
                    tempo_sel_2 = tempo_sel_2 - tempo_sel_2[onsetJoelho_sel]
                    break
            ap_sel_media = uniform_filter1d(ap_sel, size=30)
            ml_sel_media = uniform_filter1d(ml_sel, size=30)
            v_sel_media = uniform_filter1d(v_sel, size=30)
            ap_2_sel_media = uniform_filter1d(ap_2_sel, size=30)
            ml_2_sel_media = uniform_filter1d(ml_2_sel, size=30)
            v_2_sel_media = uniform_filter1d(v_2_sel, size=30)
            n1 = np.max(tempo_sel)
            n2 = np.max(tempo_sel_2)
            if n1 > n2:
                limite_tempo = n1
            else:
                limite_tempo = n2
            min_c1 = np.min(ap_sel_media[startRec:startRec+1000])
            for index,valor in enumerate(ap_sel_media):
                if valor == min_c1:
                    t_min_c1 = tempo_sel[index]
                    break
            max_c1 = np.max(ap_sel_media[index:index+1000])
            for index,valor in enumerate(ap_sel_media):
                if valor == max_c1:
                    t_max_c1 = tempo_sel[index]
                    break
            min_c2 = np.min(ap_sel_media[index:index+1000])
            for index,valor in enumerate(ap_sel_media):
                if valor == min_c2:
                    t_min_c2 = tempo_sel[index]
                    break
            max_c2 = np.max(ap_sel_media[index:index+1000])
            for index,valor in enumerate(ap_sel_media):
                if valor == max_c2:
                    t_max_c2 = tempo_sel[index]
                    break
            with col1:
                st.title("Coluna vertebral") # Cria figura com GridSpec personalizado # Cria uma figura com 3 subplots verticais
                fig = plt.figure(figsize=(12, 10))
                gs = gridspec.GridSpec(3, 2, figure=fig, wspace=0.3, hspace=0.6)
                limite = 5
                ax1 = fig.add_subplot(gs[0, 0])
                ax1.plot( tempo_sel[startRec:endRec], ap_sel[startRec:endRec], color='black', linewidth=0.8)
                ax1.plot( tempo_sel[startRec:endRec], ap_sel_media[startRec:endRec], color='red', linewidth=0.8)
                ax1.plot([t_min_c1,t_min_c1],[-4,4],"--r")
                ax1.plot([t_max_c1,t_max_c1],[-4,4],"--r")
                ax1.plot([t_min_c2,t_min_c2],[-4,4],"--r")
                ax1.plot([t_max_c2,t_max_c2],[-4,4],"--r")
                ax1.set_xlabel(r'Tempo (s)', fontsize=8)
                ax1.set_ylabel(r'Aceleração AP (m/s$^2$)', fontsize=8)
                ax1.set_xlim(-5, limite_tempo)
                ax1.set_ylim(-limite, limite)
                ax1.tick_params(axis='both', labelsize=8) # Gráfico 2: ocupa linha superior direita (metade superior)
                ax2 = fig.add_subplot(gs[1, 0])
                ax2.plot( tempo_sel[startRec:endRec], ml_sel[startRec:endRec], color='black', linewidth=0.8)
                ax2.plot(tempo_sel[startRec:endRec], ml_sel_media[startRec:endRec], color='red', linewidth=0.8)
                ax2.plot([t_min_c1,t_min_c1],[-4,4],"--r")
                ax2.plot([t_max_c1,t_max_c1],[-4,4],"--r")
                ax2.plot([t_min_c2,t_min_c2],[-4,4],"--r")
                ax2.plot([t_max_c2,t_max_c2],[-4,4],"--r")
                ax2.set_xlabel('Tempo (s)', fontsize=8)
                ax2.set_ylabel(r'Aceleração ML (m/s$^2$)', fontsize=8)
                ax2.set_xlim(-5, limite_tempo)
                ax2.set_ylim(-limite, limite)
                ax2.tick_params(axis='both', labelsize=8) # Gráfico 2: ocupa linha superior direita (metade superior)
                axv = fig.add_subplot(gs[2, 0])
                axv.plot(tempo_sel[startRec:endRec], v_sel[startRec:endRec], color='black', linewidth=0.8)
                axv.plot(tempo_sel[startRec:endRec], v_sel_media[startRec:endRec], color='red', linewidth=0.8)
                axv.plot([t_min_c1,t_min_c1],[-4,4],"--r")
                axv.plot([t_max_c1,t_max_c1],[-4,4],"--r")
                axv.plot([t_min_c2,t_min_c2],[-4,4],"--r")
                axv.plot([t_max_c2,t_max_c2],[-4,4],"--r")
                axv.set_xlabel('Tempo (s)', fontsize=8)
                axv.set_ylabel(r'Aceleração V (m/s$^2$)', fontsize=8)
                axv.set_xlim(-5, limite_tempo)
                axv.set_ylim(-limite, limite)
                axv.tick_params(axis='both', labelsize=8) # Exibe no Streamlit
                st.pyplot(fig)
            with col2:
                st.title("Joelho") # Cria figura com GridSpec personalizado # Cria uma figura com 3 subplots verticais
                fig_2 = plt.figure(figsize=(12, 10))
                gs_2 = gridspec.GridSpec(3, 2, figure=fig_2, wspace=0.3, hspace=0.6)
                ax1_2 = fig_2.add_subplot(gs_2[0, 0])
                ax1_2.plot( tempo_sel_2[startRec:endRec], ap_2_sel[startRec:endRec], color='black', linewidth=0.8)
                ax1_2.plot( tempo_sel_2[startRec:endRec], ap_2_sel_media[startRec:endRec], color='blue', linewidth=0.8)
                ax1_2.plot([t_min_c1,t_min_c1],[-4,4],"--r")
                ax1_2.plot([t_max_c1,t_max_c1],[-4,4],"--r")
                ax1_2.plot([t_min_c2,t_min_c2],[-4,4],"--r")
                ax1_2.plot([t_max_c2,t_max_c2],[-4,4],"--r")
                
                ax1_2.set_xlabel(r'Tempo (s)', fontsize=8)
                ax1_2.set_ylabel(r'Aceleração AP (m/s$^2$)', fontsize=8)
                ax1_2.set_xlim(-5, limite_tempo)
                ax1_2.set_ylim(-limite, limite)
                ax1_2.tick_params(axis='both', labelsize=8) # Gráfico 2: ocupa linha superior direita (metade superior)
                ax2_2 = fig_2.add_subplot(gs_2[1, 0])
                ax2_2.plot( tempo_sel_2[startRec:endRec], ml_2_sel[startRec:endRec], color='black', linewidth=0.8)
                ax2_2.plot(tempo_sel_2[startRec:endRec], ml_2_sel_media[startRec:endRec], color='blue', linewidth=0.8)
                ax2_2.plot([t_min_c1,t_min_c1],[-4,4],"--r")
                ax2_2.plot([t_max_c1,t_max_c1],[-4,4],"--r")
                ax2_2.plot([t_min_c2,t_min_c2],[-4,4],"--r")
                ax2_2.plot([t_max_c2,t_max_c2],[-4,4],"--r")
                ax2_2.set_xlabel('Tempo (s)', fontsize=8)
                ax2_2.set_ylabel(r'Aceleração ML (m/s$^2$)', fontsize=8)
                ax2_2.set_xlim(-5, limite_tempo)
                ax2_2.set_ylim(-limite, limite)
                ax2_2.tick_params(axis='both', labelsize=8) # Gráfico 2: ocupa linha superior direita (metade superior)
                axv_2 = fig_2.add_subplot(gs_2[2, 0])
                axv_2.plot(tempo_sel_2[startRec:endRec], v_2_sel[startRec:endRec], color='black', linewidth=0.8)
                axv_2.plot(tempo_sel_2[startRec:endRec], v_2_sel_media[startRec:endRec], color='blue', linewidth=0.8)
                axv_2.plot([t_min_c1,t_min_c1],[-4,4],"--r")
                axv_2.plot([t_max_c1,t_max_c1],[-4,4],"--r")
                axv_2.plot([t_min_c2,t_min_c2],[-4,4],"--r")
                axv_2.plot([t_max_c2,t_max_c2],[-4,4],"--r")
                axv_2.set_xlabel('Tempo (s)', fontsize=8)
                axv_2.set_ylabel(r'Aceleração V (m/s$^2$)', fontsize=8)
                axv_2.set_xlim(-5, limite_tempo)
                axv_2.set_ylim(-limite, limite)
                axv_2.tick_params(axis='both', labelsize=8) # Exibe no Streamlit
                st.pyplot(fig_2)
        if tipo_teste == "Propriocepção":
            calibracao = st.number_input('Indique o valor angular da extensão do cotovelo (em graus)', value=0.0)
            dados = st.session_state["dados"]
            st.session_state["calibracao"] = calibracao
            tempo, x_vf, y_vf, z_vf = jointSenseProcessing.processar_jps(dados, 8)
            max_val = len(tempo) # Cálculo dos ângulos
            accelAngleX = np.arctan(y_vf / np.sqrt(x_vf**2 + z_vf**2)) * 180 / math.pi
            angulo = accelAngleX + 90 
            # Calibração
            def objetivo(x):
                media_ajustada = np.mean(angulo[100:500] + x)
                return abs(media_ajustada - calibracao)
            media_baseline = np.mean(angulo[100:500])
            if media_baseline != calibracao:
                resultado = minimize_scalar(objetivo)
                angulo = angulo + resultado.x
            else:
                resultado = type('obj', (object,), {'x': 0})()
            for index,valor in enumerate(angulo[100:-1]):
                if valor > 10+calibracao:
                    t1 = index+100
                    break
            for index2,valor in enumerate(angulo[t1:-1]):
                 if valor < 10+calibracao:
                     t2 = index2+t1
                     break
            for index3,valor in enumerate(angulo[t2:-1]):
                if valor > 10+calibracao:
                    t3 = index3+t2
                    break
            for index4,valor in enumerate(angulo[t3:-1]):
                if valor < 10+calibracao:
                    t4 = index4+t3
                    break
            ref_max = np.max(angulo[t1:t2])
            for index,valor in enumerate(angulo[t1:t2]):
                if valor == ref_max:
                    t1 = t1 + index
                    t2 = t2 - index
                    break
            pos_max = np.max(angulo[t3:t4])
            for index,valor in enumerate(angulo[t3:t4]):
                if valor == pos_max:
                    t3 = t3 + index
                    t4 = t4 - index
                    break
            col1,col2,col3 = st.columns([0.2,0.8,0.2])# Cria figura com GridSpec personalizado
            with col2:
                fig = plt.figure(figsize=(8, 10))
                gs = gridspec.GridSpec(5, 4, figure=fig, hspace=0.8, wspace=0.6)
                ax1 = fig.add_subplot(gs[0:2, 0:2])
                ax1.plot(tempo, angulo, color='tomato', linewidth=0.5)
                ax1.plot([tempo[t1],tempo[t1]],[0,120],'k--')
                ax1.plot([tempo[t2],tempo[t2]],[0,120],'k--')
                ax1.plot([tempo[t3],tempo[t3]],[0,120],'k--')
                ax1.plot([tempo[t4],tempo[t4]],[0,120],'k--')
                st.pyplot(fig)
    else:
        st.info("Dados ou tipo de teste não definidos. Vá até a aba 'Importar Dados'.")
# === Página de Exportação === 
elif pagina == "📤 Exportar Resultados":
    if "dados" in st.session_state and "tipo_teste" in st.session_state:
        tipo_teste = st.session_state["tipo_teste"]
        st.subheader(f"📊 Visualização - {tipo_teste}")
        if tipo_teste == "Equilíbrio":
            dados = st.session_state["dados"]
            startRec, endRec, filter = st.session_state["intervalo"]
            rms_ml, rms_ap, total_deviation, ellipse_area, avg_x, avg_y, width, height, angle, direction = balanceProcessing.processar_equilibrio( dados, startRec, endRec, 1, 1, filter)
            tempo_sel, ml_sel, ap_sel, freqs_sel, psd_ml_sel, psd_ap_sel = balanceProcessing.processar_equilibrio( dados, startRec, endRec, 1, 0, filter)
            planar_dev = np.rad2deg( np.sqrt(np.std(ml_sel)**2 + np.std(ap_sel)**2))
            sway_dir = np.rad2deg(np.arctan2(np.std(ap_sel), np.std(ml_sel)))
            total_power_ml = trapezoid(psd_ml_sel, freqs_sel)
            total_power_ap = trapezoid(psd_ap_sel, freqs_sel)
            bands = [(0, 0.5), (0.5, 2), (2, np.inf)]
            energy_ml = [trapezoid(psd_ml_sel[(freqs_sel >= b[0]) & (freqs_sel < b[1])], freqs_sel[( freqs_sel >= b[0]) & (freqs_sel < b[1])]) for b in bands]
            energy_ap = [trapezoid(psd_ap_sel[(freqs_sel >= b[0]) & (freqs_sel < b[1])], freqs_sel[( freqs_sel >= b[0]) & (freqs_sel < b[1])]) for b in bands]
            centroid_ml = trapezoid( freqs_sel * psd_ml_sel, freqs_sel) / total_power_ml
            centroid_ap = trapezoid( freqs_sel * psd_ap_sel, freqs_sel) / total_power_ap
            mode_ml = freqs_sel[np.argmax(psd_ml_sel)]
            mode_ap = freqs_sel[np.argmax(psd_ap_sel)]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(label=r"RMS ML (m/s$^2$)", value=round(rms_ml, 4))
                st.metric(label=r"RMS AP (m/s$^2$)", value=round(rms_ap, 4))
                st.metric(label=r"Desvio total (m/s$^2$)", value=round(total_deviation, 4))
                st.metric(label=r"Área da elipse (m$^2$/s$^4$)", value=round(ellipse_area, 4))
                st.metric(label="Desvio planar (graus)", value=round(planar_dev, 4))
                st.metric(label="Direção da oscilação (graus)", value=round(sway_dir, 4))
            with col2:
                st.metric(label="Centróide ML (Hz)", value=round(centroid_ml, 4))
                st.metric(label="Centróide AP (Hz)", value=round(centroid_ap, 4))
            with col3:
                st.metric(label=r"Energia ML 0-0.5 Hz (m/s$^2$)", value=round((10**5)*energy_ml[0], 3))
                st.metric(label=r"Energia ML 0.5-2 Hz (m/s$^2$)", value=round((10**5)*energy_ml[1], 3))
                st.metric(label=r"Energia ML 2-8 Hz (m/s$^2$)", value=round((10**5)*energy_ml[2], 3))
            with col4:
                st.metric(label=r"Energia AP 0-0.5 Hz (m/s$^2$)", value=round((10**5)*energy_ap[0], 3))
                st.metric(label=r"Energia AP 0.5-2 Hz (m/s$^2$)", value=round((10**5)*energy_ap[1], 3))
                st.metric(label=r"Energia AP 2-8 Hz (m/s$^2$)", value=round((10**5)*energy_ap[2], 3))
            resultado_txt = "Variável\tValor\n" # Cabeçalho com tabulação # Lista de pares (nome, valor) 
            variaveis = [("RMS ML", round(rms_ml, 4)),
                         ("RMS AP", round(rms_ap, 4)),
                         ("Desvio total", round(total_deviation, 4)),
                         ("Área da elipse", round(ellipse_area, 4)),
                         ("Desvio planar", round(planar_dev, 4)),
                         ("Direção da oscilação", round(sway_dir, 4)),
                         ("Centróide ML", round(centroid_ml, 4)),
                         ("Centróide AP", round(centroid_ap, 4)),
                         ("Energia ML 0–0.5 Hz", round((10**5)*energy_ml[0], 8)),
                         ("Energia ML 0.5–2 Hz", round((10**5)*energy_ml[1], 8)),
                         ("Energia ML 2–8 Hz", round((10**5)*energy_ml[2], 8)),
                         ("Energia AP 0–0.5 Hz", round((10**5)*energy_ap[0], 8)),
                         ("Energia AP 0.5–2 Hz", round((10**5)*energy_ap[1], 8)),
                         ("Energia AP 2–8 Hz", round((10**5)*energy_ap[2], 8)),] # Adiciona linha por linha 
            for nome, valor in variaveis:
                resultado_txt += f"{nome}\t{valor}\n"
            st.download_button(label="📄 Exportar resultados (.txt)", data=resultado_txt, file_name="resultados_analise_postural.txt", mime="text/plain" )
        if tipo_teste == "Salto":
            dados = st.session_state["dados"]
            tempo, salto, startJump, endJump, altura, tempo_voo, m1, m2, veloc, desloc, istart, iend = jumpProcessing.processar_salto( dados, 8)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(label="Tempo de vôo (s)", value=round(tempo_voo, 4))
                st.metric(label="Altura (m)", value=round(altura, 4))
            with col2:
                st.metric( label="Velocidade de decolagem (m/s)", value=round(veloc, 4))
        if tipo_teste == "TUG":
            baseline_onset = st.session_state["baseline_onset"]
            baseline_offset = st.session_state["baseline_offset"]
            col1, col2, col3 = st.columns([0.4,0.8,0.6])
            dados_acc = st.session_state["dados_acc"]
            dados_gyro = st.session_state["dados_gyro"]
            t_novo_acc, v_acc, ml_acc, z_acc_filtrado, norma_acc_filtrado, t_novo_gyro, v_gyro, ml_gyro, z_gyro_filtrado, norma_gyro_filtrado,start_test,stop_test,idx,idx_ml,idx_acc_ap,idx_acc_v,duration = tugProcessing.processar_tug(dados_acc,dados_gyro,2,1.25,baseline_onset,baseline_offset)
            vertical_squared = np.sqrt(v_gyro**2)
            lat1 = idx[1][0]
            lat2 = idx[1][1]
            amp1 = vertical_squared[idx[0][0]]
            amp2 = vertical_squared[idx[0][1]]
            if lat1 > lat2:
                G1_lat = lat2
                G1_amp = amp2
                G2_lat = lat1
                G2_amp = amp1
            else:
                G1_lat = lat1
                G1_amp = amp1
                G2_lat = lat2
                G2_amp = amp2
            ml_squared = np.sqrt(ml_gyro**2)
            lat1 = idx_ml[1][0]
            lat2 = idx_ml[1][1]
            amp1 = ml_squared[idx_ml[0][0]]
            amp2 = ml_squared[idx_ml[0][1]]
            if lat1 > lat2:
                G0_lat = lat2
                G0_amp = amp2
                G4_lat = lat1
                G4_amp = amp1
            else: 
                G0_lat = lat1
                G0_amp = amp1
                G4_lat = lat2
                G4_amp = amp2
            acc_ap_squared = np.sqrt(z_acc_filtrado**2)
            lat1 = idx_acc_ap[1][0]
            lat2 = idx_acc_ap[1][1]
            amp1 = acc_ap_squared[idx_acc_ap[0][0]]
            amp2 = acc_ap_squared[idx_acc_ap[0][1]]
            if lat1 > lat2:
                A1_lat = lat2
                A1_amp = amp2
                A2_lat = lat1
                A2_amp = amp1
            else:
                A1_lat = lat1
                A1_amp = amp1
                A2_lat = lat2
                A2_amp = amp2
            acc_v_squared = np.sqrt(v_acc**2)
            lat1 = idx_acc_v[1][0]
            lat2 = idx_acc_v[1][1]
            amp1 = acc_v_squared[idx_acc_v[0][0]]
            amp2 = acc_v_squared[idx_acc_v[0][1]]
            if lat1 > lat2:
                A1v_lat = lat2
                A1v_amp = amp2
                A2v_lat = lat1
                A2v_amp = amp1
            else:
                A1v_lat = lat1
                A1v_amp = amp1
                A2v_lat = lat2
                A2v_amp = amp2
            with col1:
                st.metric(label=r"Duração do teste (s)", value=round(stop_test-start_test, 4))
                st.metric(label=r"Duração para levantar (s)", value=round(G0_lat-start_test, 4))
                st.metric(label=r"Duração da caminhada de ida (s)", value=round(G1_lat-G0_lat, 4))
                st.metric(label=r"Duração da caminhada de volta (s)", value=round(G2_lat-G1_lat, 4))
                st.metric(label=r"Duração para sentar (s)", value=round(stop_test-G2_lat, 4))
            with col2:
                st.metric(label=r"Tempo para o pico de velocidade angular na transição de sentado para de pé (s)", value=round(G0_lat-start_test, 4))
                st.metric(label=r"Tempo para o pico de aceleração AP na transição de sentado para de pé (s)", value=round(A1_lat-start_test, 4))
                st.metric(label=r"Tempo para o pico de aceleração V na transição de sentado para de pé (s)", value=round(A1v_lat-start_test, 4))
                st.metric(label=r"Tempo para o pico de velocidade angular no giro em 3 m (s)", value=round(G1_lat-start_test, 4))
                st.metric(label=r"Tempo para o pico de velocidade angular no giro em 6 m (s)", value=round(G2_lat-start_test, 4))
                st.metric(label=r"Tempo para o pico de velocidade angular na transição de pé para sentado (s)", value=round(G2_lat-start_test, 4))
                st.metric(label=r"Tempo para o pico de aceleração AP na transição de pé para sentado (s)", value=round(A2_lat-start_test, 4))
                st.metric(label=r"Tempo para o pico de aceleração V na transição de pé para sentado (s)", value=round(A2v_lat-start_test, 4))
            with col3:
                st.metric(label=r"Velocidade angular máxima na transição de sentado para de pé (rad/s)", value=round(G0_amp, 4))
                st.metric(label=r"Aceleração máxima AP na transição de sentado para de pé (m/s2)", value=round(A1_amp, 4))
                st.metric(label=r"Aceleração máxima V na transição de sentado para de pé (m/s2)", value=round(A1v_amp, 4))
                st.metric(label=r"Velocidade angular máxima no giro em 3 m (s)", value=round(G1_amp, 4))
                st.metric(label=r"Velocidade angular máxima no giro em 6 m (s)", value=round(G2_amp, 4))
                st.metric(label=r"Aceleração máxima AP na transição de sentado para pé (s)", value=round(A2_amp, 4))
                st.metric(label=r"Aceleração máxima V na transição de sentado para pé (s)", value=round(A2v_amp, 4))
            resultado_txt = "Variável\tValor\n" # Cabeçalho com tabulação # Lista de pares (nome, valor)
            variaveis = [("Duração do teste (s)", round(stop_test-start_test, 4)),
                         ("Duração para levantar (s)", round(G0_lat-start_test, 4)),
                         ("Duração da caminhada de ida (s)",round(G1_lat-G0_lat, 4)),
                         ("Duração da caminhada de volta (s)",round(G2_lat-G1_lat, 4)),
                         ("Duração para sentar (s)",round(stop_test-G2_lat, 4)),
                         ("Tempo para o pico de velocidade angular na transição de sentado para de pé (s)",round(G0_lat-start_test, 4)),
                         ("Tempo para o pico de aceleração AP na transição de sentado para de pé (s)",round(A1_lat-start_test, 4)),
                         ("Tempo para o pico de aceleração V na transição de sentado para de pé (s)",round(A1v_lat-start_test, 4)),
                         ("Tempo para o pico de velocidade angular no giro em 3 m (s)",round(G1_lat-start_test, 4)),
                         ("Tempo para o pico de velocidade angular no giro em 6 m (s)",round(G2_lat-start_test, 4)),
                         ("Tempo para o pico de velocidade angular na transição de pé para sentado (s)",round(A2_lat-start_test, 4)),
                         ("Tempo para o pico de aceleração AP na transição de pé para sentado (s)",round(A2_lat-start_test, 4)),
                         ("Tempo para o pico de aceleração V na transição de pé para sentado (s)",round(A2v_lat-start_test, 4)),
                         ("Velocidade angular máxima na transição de sentado para de pé (rad/s)",round(G0_amp, 4)),
                         ("Aceleração máxima AP na transição de sentado para de pé (m/s2)",round(A1_amp, 4)),
                         ("Aceleração máxima V na transição de sentado para de pé (m/s2)",round(A1v_amp, 4)),
                         ("Velocidade angular máxima no giro em 3 m (s)",round(G1_amp, 4)),
                         ("Velocidade angular máxima no giro em 6 m (s)",round(G2_amp, 4)),
                         ("Aceleração máxima AP na transição de sentado para pé (s)",round(A2_amp, 4)),
                         ("Aceleração máxima V na transição de sentado para pé (s)",round(A2v_amp, 4))] # Adiciona linha por linha 
            for nome, valor in variaveis:
                resultado_txt += f"{nome}\t{valor}\n"
            st.download_button(label="📄 Exportar resultados (.txt)", data=resultado_txt, file_name="resultados_analise_iTUG.txt", mime="text/plain" )    
                
        if tipo_teste == "Y test":
            dados = st.session_state["dados_acc_coluna"]
            dados2 = st.session_state["dados_acc_joelho"]
            tempo, ml, ap, v= ytestProcessing.processar_ytest1(dados,8)
            max_val = 5000
            col1, col2, col3 = st.columns(3)
            with col1:
                startRec = st.number_input( 'Indique o início do registro', value=0, step=1, max_value=max_val)
            with col2:
                endRec = st.number_input( 'Indique o final do registro', value=max_val, step=1, max_value=max_val)
            with col3:
                filter = st.number_input( 'Indique o filtro passa-baixa', value=8.0, step=0.1, max_value=40.0)
            showRec = st.checkbox('Mostrar o dado original', value=True)
            tempo, ml, ap, v= ytestProcessing.processar_ytest1(dados[0:len(dados)-10],filter)
            tempo_2, ml_2, ap_2, v_2= ytestProcessing.processar_ytest2(dados2[0:len(dados2)-10],filter)
            col1, col2 = st.columns(2)
            tempo_sel, ml_sel, ap_sel, v_sel = ytestProcessing.processar_ytest1(dados[startRec:endRec], filter)
            tempo_sel_2, ml_2_sel, ap_2_sel, v_2_sel = ytestProcessing.processar_ytest2(dados2[startRec:endRec], filter)
            picoSaltoCintura = np.max(v[0:1000])
            for index,valor in enumerate(v):
                if valor == picoSaltoCintura:
                    onsetCintura = index
                    tempo = tempo - tempo[onsetCintura]
                    break
            picoSaltoJoelho = np.max(v_2[0:1000])
            for index,valor in enumerate(v_2):
                if valor == picoSaltoJoelho:
                    onsetJoelho = index
                    tempo_2 = tempo_2 - tempo_2[onsetJoelho]
                    break
            picoSaltoCintura_sel = np.max(v_sel[0:1000])
            for index,valor in enumerate(v_sel):
                if valor == picoSaltoCintura_sel:
                    onsetCintura_sel = index
                    tempo_sel = tempo_sel - tempo_sel[onsetCintura_sel]
                    break
            picoSaltoJoelho_sel = np.max(v_2_sel[0:1000])
            for index,valor in enumerate(v_2_sel):
                if valor == picoSaltoJoelho_sel:
                    onsetJoelho_sel = index
                    tempo_sel_2 = tempo_sel_2 - tempo_sel_2[onsetJoelho_sel]
                    break
            ap_sel_media = uniform_filter1d(ap_sel, size=30)
            ml_sel_media = uniform_filter1d(ml_sel, size=30)
            v_sel_media = uniform_filter1d(v_sel, size=30)
            ap_2_sel_media = uniform_filter1d(ap_2_sel, size=30)
            ml_2_sel_media = uniform_filter1d(ml_2_sel, size=30)
            v_2_sel_media = uniform_filter1d(v_2_sel, size=30)
            n1 = np.max(tempo_sel)
            n2 = np.max(tempo_sel_2)
            if n1 > n2:
                limite_tempo = n1
            else:
                limite_tempo = n2
            
            min_c1 = np.min(ap_sel_media[startRec:startRec+1000])
            for index,valor in enumerate(ap_sel_media):
                if valor == min_c1:
                    t_min_c1 = tempo_sel[index]
                    break
            for index2,valor in enumerate(tempo_sel_2):
                if valor > t_min_c1:
                    break
            dados = ap_2_sel_media[index2-50:index2]        
            ampC1_ap_pre = np.sqrt(np.mean(np.square(dados)))
            dados = ap_2_sel_media[index2:index2+50]        
            ampC1_ap_pos = np.sqrt(np.mean(np.square(dados)))

            dados = ml_2_sel_media[index2-50:index2]        
            ampC1_ml_pre = np.sqrt(np.mean(np.square(dados)))
            dados = ml_2_sel_media[index2:index2+50]        
            ampC1_ml_pos = np.sqrt(np.mean(np.square(dados)))
            
            max_c1 = np.max(ap_sel_media[index:index+1000])
            for index,valor in enumerate(ap_sel_media):
                if valor == max_c1:
                    t_max_c1 = tempo_sel[index]
                    break
            for index3,valor in enumerate(tempo_sel_2):
                if valor > t_max_c1:
                    break
            dados = ap_2_sel_media[index3-50:index3]        
            ampC2_ap_pre = np.sqrt(np.mean(np.square(dados)))
            dados = ap_2_sel_media[index3:index3+50]        
            ampC2_ap_pos = np.sqrt(np.mean(np.square(dados)))

            dados = ml_2_sel_media[index3-50:index3]        
            ampC2_ml_pre = np.sqrt(np.mean(np.square(dados)))
            dados = ml_2_sel_media[index3:index3+50]        
            ampC2_ml_pos = np.sqrt(np.mean(np.square(dados)))
                       
            min_c2 = np.min(ap_sel_media[index:index+1000])
            for index,valor in enumerate(ap_sel_media):
                if valor == min_c2:
                    t_min_c2 = tempo_sel[index]
                    break
            for index4,valor in enumerate(tempo_sel_2):
                if valor > t_min_c2:
                    break
            dados = ap_2_sel_media[index4-50:index4]        
            ampC3_ap_pre = np.sqrt(np.mean(np.square(dados)))
            dados = ap_2_sel_media[index4:index4+50]        
            ampC3_ap_pos = np.sqrt(np.mean(np.square(dados)))

            dados = ml_2_sel_media[index4-50:index4]        
            ampC3_ml_pre = np.sqrt(np.mean(np.square(dados)))
            dados = ml_2_sel_media[index4:index4+50]        
            ampC3_ml_pos = np.sqrt(np.mean(np.square(dados)))
            
            max_c2 = np.max(ap_sel_media[index:index+1000])
            for index,valor in enumerate(ap_sel_media):
                if valor == max_c2:
                    t_max_c2 = tempo_sel[index]
                    break
            for index5,valor in enumerate(tempo_sel_2):
                if valor > t_max_c2:
                    break        
            dados = ap_2_sel_media[index5-50:index5]        
            ampC4_ap_pre = np.sqrt(np.mean(np.square(dados)))
            dados = ap_2_sel_media[index5:index5+50]        
            ampC4_ap_pos = np.sqrt(np.mean(np.square(dados)))   

            dados = ml_2_sel_media[index5-50:index5]        
            ampC4_ml_pre = np.sqrt(np.mean(np.square(dados)))
            dados = ml_2_sel_media[index5:index5+50]        
            ampC4_ml_pos = np.sqrt(np.mean(np.square(dados)))
                    
            col1,col2,col3,col4 = st.columns(4)
            with col1:
                st.metric(label=r"Amplitude de C1 (m/s2)", value=round(min_c1, 4))
                st.metric(label=r"Tempo de C1 (s)", value=round(t_min_c1, 4))
                st.metric(label=r"Amplitude pré-C1 Joelho AP (m/s2)", value=round(ampC1_ap_pre, 4))
                st.metric(label=r"Amplitude pós-C1 Joelho AP (m/s2)", value=round(ampC1_ap_pos, 4))
                st.metric(label=r"Amplitude pré-C1 Joelho ML (m/s2)", value=round(ampC1_ml_pre, 4))
                st.metric(label=r"Amplitude pós-C1 Joelho ML (m/s2)", value=round(ampC1_ml_pos, 4))
            with col2:
                st.metric(label=r"Amplitude de C2 (m/s2)", value=round(max_c1, 4))
                st.metric(label=r"Tempo de C2 (s)", value=round(t_max_c1, 4))
                st.metric(label=r"Amplitude pré-C2 Joelho AP (m/s2)", value=round(ampC2_ap_pre, 4))
                st.metric(label=r"Amplitude pós-C2 Joelho AP (m/s2)", value=round(ampC2_ap_pos, 4))
                st.metric(label=r"Amplitude pré-C2 Joelho ML (m/s2)", value=round(ampC2_ml_pre, 4))
                st.metric(label=r"Amplitude pós-C2 Joelho ML (m/s2)", value=round(ampC2_ml_pos, 4))
            with col3:
                st.metric(label=r"Amplitude de C3 (m/s2)", value=round(min_c2, 4))
                st.metric(label=r"Tempo de C3 (s)", value=round(t_min_c2, 4))
                st.metric(label=r"Amplitude pré-C3 Joelho AP (m/s2)", value=round(ampC3_ap_pre, 4))
                st.metric(label=r"Amplitude pós-C3 Joelho AP (m/s2)", value=round(ampC3_ap_pos, 4))
                st.metric(label=r"Amplitude pré-C3 Joelho ML (m/s2)", value=round(ampC3_ml_pre, 4))
                st.metric(label=r"Amplitude pós-C3 Joelho ML (m/s2)", value=round(ampC3_ml_pos, 4))
            with col4:
                st.metric(label=r"Amplitude de C4 (m/s2)", value=round(max_c2, 4))
                st.metric(label=r"Tempo de C4 (s)", value=round(t_max_c2, 4))
                st.metric(label=r"Amplitude pré-C4 Joelho AP (m/s2)", value=round(ampC4_ap_pre, 4))
                st.metric(label=r"Amplitude pós-C4 Joelho AP (m/s2)", value=round(ampC4_ap_pos, 4))
                st.metric(label=r"Amplitude pré-C4 Joelho ML (m/s2)", value=round(ampC4_ml_pre, 4))
                st.metric(label=r"Amplitude pós-C4 Joelho ML (m/s2)", value=round(ampC4_ml_pos, 4))
            resultado_txt = "Variável\tValor\n" # Cabeçalho com tabulação # Lista de pares (nome, valor)
            variaveis = [("Amplitude de C1 (m/s2)", round(min_c1, 4)),
                         ("Tempo de C1 (s)", round(t_min_c1, 4)),
                         ("Amplitude de C2 (m/s2)", round(max_c1, 4)),
                         ("Tempo de C2 (s)", round(t_max_c1, 4)),
                         ("Amplitude de C3 (m/s2)", round(min_c2, 4)),
                         ("Tempo de C3 (s)", round(t_min_c2, 4)),
                         ("Amplitude de C4 (m/s2)", round(max_c2, 4)),
                         ("Tempo de C4 (s)", round(t_max_c2, 4)),
                         ("Amplitude pré-C1 Joelho AP (m/s2)",round(ampC1_ap_pre, 4)),
                         ("Amplitude pós-C1 Joelho AP (m/s2)",round(ampC1_ap_pos, 4)),
                         ("Amplitude pré-C2 Joelho AP (m/s2)",round(ampC2_ap_pre, 4)),
                         ("Amplitude pós-C2 Joelho AP (m/s2)",round(ampC2_ap_pos, 4)),
                         ("Amplitude pré-C3 Joelho AP (m/s2)",round(ampC3_ap_pre, 4)),
                         ("Amplitude pós-C3 Joelho AP (m/s2)",round(ampC3_ap_pos, 4)),
                         ("Amplitude pré-C4 Joelho AP (m/s2)",round(ampC4_ap_pre, 4)),
                         ("Amplitude pós-C4 Joelho AP (m/s2)",round(ampC4_ap_pos, 4)),
                         ("Amplitude pré-C1 Joelho ML (m/s2)",round(ampC1_ml_pre, 4)),
                         ("Amplitude pós-C1 Joelho ML (m/s2)",round(ampC1_ml_pos, 4)),
                         ("Amplitude pré-C2 Joelho ML (m/s2)",round(ampC2_ml_pre, 4)),
                         ("Amplitude pós-C2 Joelho ML (m/s2)",round(ampC2_ml_pos, 4)),
                         ("Amplitude pré-C3 Joelho ML (m/s2)",round(ampC3_ml_pre, 4)),
                         ("Amplitude pós-C3 Joelho ML (m/s2)",round(ampC3_ml_pos, 4)),
                         ("Amplitude pré-C4 Joelho ML (m/s2)",round(ampC4_ml_pre, 4)),
                         ("Amplitude pós-C4 Joelho ML (m/s2)",round(ampC4_ml_pos, 4)),] # Adiciona linha por linha 
            for nome, valor in variaveis:
                resultado_txt += f"{nome}\t{valor}\n"
            st.download_button(label="📄 Exportar resultados (.txt)", data=resultado_txt, file_name="resultados_analise_postural.txt", mime="text/plain" )
        if tipo_teste == "Propriocepção":
            calibracao = st.session_state["calibracao"]
            dados = st.session_state["dados"]
            tempo, x_vf, y_vf, z_vf = jointSenseProcessing.processar_jps(dados, 8)
            max_val = len(tempo) # Cálculo dos ângulos
            accelAngleX = np.arctan(y_vf / np.sqrt(x_vf**2 + z_vf**2)) * 180 / math.pi
            angulo = accelAngleX + 90 # Calibração
            def objetivo(x):
                media_ajustada = np.mean(angulo[100:500] + x)
                return abs(media_ajustada - calibracao)
            media_baseline = np.mean(angulo[100:500])
            if media_baseline != calibracao:
                resultado = minimize_scalar(objetivo)
                angulo = angulo + resultado.x
            else:
                resultado = type('obj', (object,), {'x': 0})()
            for index,valor in enumerate(angulo[100:-1]):
                if valor > 10+calibracao:
                    t1 = index+100
                    break
            for index2,valor in enumerate(angulo[t1:-1]):
                if valor < 10+calibracao:
                    t2 = index2+t1
                    break
            for index3,valor in enumerate(angulo[t2:-1]):
                if valor > 10+calibracao:
                    t3 = index3+t2
                    break
            for index4,valor in enumerate(angulo[t3:-1]):
                if valor < 10+calibracao:
                    t4 = index4+t3
                    break
            ref_max = np.max(angulo[t1:t2])
            for index,valor in enumerate(angulo[t1:t2]):
                if valor == ref_max:
                    t1 = t1 + index
                    t2 = t2 - index
                    break
            pos_max = np.max(angulo[t3:t4])
            for index,valor in enumerate(angulo[t3:t4]):
                if valor == pos_max:
                    t3 = t3 + index
                    t4 = t4 - index
                    break
            Angulacao_referencia = np.mean(angulo[t1:t2])
            Angulacao_posicionamento = np.mean(angulo[t3:t4])
            st.metric(label=r"Ângulo de referências (graus)", value=round(Angulacao_referencia, 4))
            st.metric(label=r"Ângulo de posicionamento (graus)", value=round(Angulacao_posicionamento, 4))
            resultado_txt = "Variável\tValor\n"  # Cabeçalho com tabulação

            variaveis = [
                ("Angulo médio de referência (graus)", round(Angulacao_referencia, 4)),
                ("Angulo médio de posicionamento (graus)", round(Angulacao_posicionamento, 4))
            ]
            
            for nome, valor in variaveis:
                resultado_txt += f"{nome}\t{valor}\n"
            
            st.download_button(
                label="📄 Exportar resultados (.txt)",
                data=resultado_txt,
                file_name="resultados_propriocepcao.txt",
                mime="text/plain"
            )
elif pagina == "📖 Referências bibliográficas":
    html = dedent(""" <div style="text-align: justify; font-size: 1.1rem; line-height: 1.6; color: #333333; max-width: 900px; margin: auto; background-color: rgba(255,200,255,0.6); padding: 20px; border-radius: 8px;"> 
    <p> Artigos que utilizaram aplicativos desenvolvidos no projeto Momentum:
    </p>
    <a href="https://www.mdpi.com/1424-8220/24/9/2918" target="_blank" style="color:#1E90FF; text-decoration:none;">1. SANTOS, P. S. A. ; SANTOS, E. G. R. ; MONTEIRO, L. C. P. ; SANTOS-LOBATO, B. L. ; PINTO, G. H. L. ; BELGAMO, A. ; ANDRÉ DOS SANTOS, CABRAL ; COSTA E SILVA, A. A ; CALLEGARI, B. ; SOUZA, Givago da Silva . The hand tremor spectrum is modified by the inertial sensor mass during lightweight wearable and smartphone-based assessment in healthy young subjects. Scientific Reports, v. 12, p. 01, 2022.</a></p>. 
    <a href="https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2023.1277408/full" target="_blank" style="color:#1E90FF; text-decoration:none;">2. RODRIGUES, L. A. ; SANTOS, E. G. R. ; SANTOS, P. S. A. ; IGARASHI, Y. ; OLIVEIRA, L. K. R. ; PINTO, G. H. L. ; SANTOS-LOBATO, B. L. ; CABRAL, A. S. ; BELGAMO, A. ; COSTA E SILVA, A. A ; CALLEGARI, B. ; Souza, G. S. . Wearable Devices and Smartphone Inertial Sensors for Static Balance Assessment: A Concurrent Validity Study in Young Adult Population. Journal Of Personalized Medicine, v. 1, p. 1-1, 2022.</a></p>
    <a href="https://www.scielo.br/j/aabc/a/7z5HDVZKYVMxfWm8HxcJqZG/?lang=en&format=pdf" target="_blank" style="color:#1E90FF; text-decoration:none;">3. CORREA, B. D. C. ; SANTOS, E. G. R. ; BELGAMO, A. ; PINTO, G. H. L. ; XAVIER, S. S. ; DIAS, A. R. N. ; PARANHOS, A. C. M. ; ANDRÉ DOS SANTOS, CABRAL ; CALLEGARI, B. ; COSTA E SILVA, A. A. ; QUARESMA, J. A. S. ; FALCAO, L. F. M. ; SOUZA, GIVAGO S. . SMARTPHONE-BASED EVALUATION OF STATIC BALANCE AND MOBILITY IN LONG LASTING COVID-19 PATIENTS. Frontiers in Neurology, v. 1, p. 1, 2023.</a>.</p> 
    <a href="https://www.scielo.br/j/aabc/a/7z5HDVZKYVMxfWm8HxcJqZG/?lang=en&format=pdf" target="_blank" style="color:#1E90FF; text-decoration:none;">4. FURTADO, E. C. S. ; AZEVEDO, Y. S. ; GALHARDO, D. R. ; MIRANDA, I. P. C. ; OLIVEIRA, M. E. C. ; NEVES, P. F. M. ; MONTE, L. B. ; NUNES, E. F. C. ; FERREIRA, E. A. G. ; CALLEGARI, B. ; SOUZA, G.S. ; MELO NETO, J. S. . The weeks of gestation age influence the myoelectric activity of the pelvic floor muscles, plantar contact and functional mobility in high-risk pregnant women? a cross-sectional study. SENSORS, v. 1, p. 1, 2024.</a>.</p>
    <a href="https://www.scielo.br/j/aabc/a/7z5HDVZKYVMxfWm8HxcJqZG/?lang=en&format=pdf" target="_blank" style="color:#1E90FF; text-decoration:none;">5. SANTOS, T. T. S. ; MARQUES, A. P. ; MONTEIRO, L. C. P. ; SANTOS, E. G. R. ; PINTO, G. H. L. ; BELGAMO, A. ; COSTA E SILVA, A. A. ; CABRAL, A. S. ; KULIŚ ; GAJEWSKI, J. ; Souza, G. S. ; SILVA, T. J. ; COSTA, W. T. A. ; SALOMAO, R. C. ; CALLEGARI, B. . Intra and Inter-Device Reliabilities of the Instrumented Timed-Up and Go Test Using Smartphones in Young Adult Population. SENSORS, v. 24, p. 2918, 2024.</a>.</p> 
    <a href="https://www.scielo.br/j/aabc/a/7z5HDVZKYVMxfWm8HxcJqZG/?lang=en&format=pdf" target="_blank" style="color:#1E90FF; text-decoration:none;">6. FERNANDES, T. F. ; CÔRTES, M. I. T. ; PENA, F. ; SANTOS, E. G. R. ; PINTO, G. H. L. ; BELGAMO, A. ; COSTA E SILVA, A. A. ; ANDRÉ DOS SANTOS, CABRAL ; CALLEGARI, B. ; Souza, G. S. . Smartphone-based evaluation of static balance and mobility in type 2 Diabetes. ANAIS DA ACADEMIA BRASILEIRA DE CIÊNCIAS, v. 96, p. 1-1, 2024.</a>.</p> 
    <a href="https://www.scielo.br/j/aabc/a/7z5HDVZKYVMxfWm8HxcJqZG/?lang=en&format=pdf" target="_blank" style="color:#1E90FF; text-decoration:none;">7. NASCIMENTO, A. Q. ; NAGATA, L. A. R. ; ALMEIDA, M. T. ; COSTA, V. L. S. ; MARIN, A. B. R. ; TAVARES, V. B. ; ISHAK, G. ; CALLEGARI, B. ; SANTOS, E. G. R. ; SOUZA, GIVAGO SILVA ; MELO NETO, J. S. . Smartphone-based inertial measurements during Chester step test as a predictor of length of hospital stay in abdominopelvic cancer postoperative period: a prospective cohort study. World Journal of Surgical Oncology, v. 22, p. 71-1, 2024.</a>.</p> 
    <a href="https://www.scielo.br/j/aabc/a/7z5HDVZKYVMxfWm8HxcJqZG/?lang=en&format=pdf" target="_blank" style="color:#1E90FF; text-decoration:none;">8. FERREIRA, E. C. V. ; MARQUES, A. P. ; KULIS, S. ; GAJEWSKI, J. ; MORAES, A. A. C. ; DUARTE, M. B. ; ALMEIDA, G. C. S. ; SANTOS, E. G. R. ; PINTO, G. H. L. ; ANDRÉ DOS SANTOS, CABRAL ; Souza, Gilvago Silva ; COSTA E SILVA, A. A ; CALLEGARI, B. . Validity And Reliability of a Mobile Device Application for Assessing Motor Performance in the 30-Second Sit-To-Stand Test. IEEE Access, p. 1-1, 2025.</a>.</p> 
    <a href="https://www.scielo.br/j/aabc/a/7z5HDVZKYVMxfWm8HxcJqZG/?lang=en&format=pdf" target="_blank" style="color:#1E90FF; text-decoration:none;">9. AZEVEDO, L. S. ; FEITOSA JR, N. Q. ; SANTOS, E. G. R. ; ALVAREZ, M. A. M. ; NORAT, L. A. X. ; BOTELHO, G. I. S. ; BELGAMO, A. ; PINTO, G. H. L. ; SANTANA DE CASTRO, KETLIN JAQUELLINE ; CALLEGARI, B. ; SILVA, A. A. C. E. ; SALOMAO, R. C. ; ANDRÉ DOS SANTOS, CABRAL ; ROSA, A. A. M. ; Silva Souza, Givago . Assessing static balance control improvement following cataract surgery using a smartphone. Digital Health, v. 11, p. 1-1, 2025.</a>.</p> 
    <a href="https://www.scielo.br/j/aabc/a/7z5HDVZKYVMxfWm8HxcJqZG/?lang=en&format=pdf" target="_blank" style="color:#1E90FF; text-decoration:none;">10. DUARTE, M. B. ; MORAES, A. A. C. ; FERREIRA, E. V. ; ALMEIDA, G. C. S. ; SANTOS, E. G. R. ; PINTO, G. H. L. ; OLIVEIRA, P. R. ; AMORIM, C. F. ; ANDRÉ DOS SANTOS, CABRAL ; SAUNIER, G. J. A. ; COSTA E SILVA, A. A. ; SOUZA, GIVAGO S. ; CALLEGARI, B. . Validity and reliability of a smartphone-based assessment for anticipatory and compensatory postural adjustments during predictable perturbations. GAIT & POSTURE, v. 96, p. 9-17, 2022.</a>.</p> 
    <a href="https://www.scielo.br/j/aabc/a/7z5HDVZKYVMxfWm8HxcJqZG/?lang=en&format=pdf" target="_blank" style="color:#1E90FF; text-decoration:none;">11. MORAES, A. A. C. ; DUARTE, M. B. ; FERREIRA, E. V. ; ALMEIDA, G. C. S. ; SANTOS, E. G. R. ; PINTO, G. H. L. ; OLIVEIRA, P. R. ; AMORIM, C. F. ; ANDRÉ DOS SANTOS, CABRAL ; COSTA E SILVA, A. A. ; Souza, G. S. ; CALLEGARI, B. . Validity and reliability of smartphone app for evaluating postural adjustments during step initiation. SENSORS, v. 1, p. 1, 2022.</a>.</p> 
    <a href="https://www.scielo.br/j/aabc/a/7z5HDVZKYVMxfWm8HxcJqZG/?lang=en&format=pdf" target="_blank" style="color:#1E90FF; text-decoration:none;">12. MORAES, A. A. C. ; DUARTE, M. B. ; SANTOS, E. J. M. ; ALMEIDA, G. C. S. ; ANDRÉ DOS SANTOS, CABRAL ; COSTA E SILVA, A. A. ; GARCEZ, D. R. ; GIVAGO DA SILVA, SOUZA ; CALLEGARI, B. . Comparison of inertial records during anticipatory postural adjustments obtained with devices of different masses. PeerJ, v. 11, p. e15627, 2023.</a>.</p> 
    <a href="https://www.scielo.br/j/aabc/a/7z5HDVZKYVMxfWm8HxcJqZG/?lang=en&format=pdf" target="_blank" style="color:#1E90FF; text-decoration:none;">13. BRITO, F. A. C. ; MONTEIRO, L. C. P. ; SANTOS, E. G. R. ; LIMA, R. C. ; SANTOS-LOBATO, B. L. ; ANDRÉ DOS SANTOS, CABRAL ; CALLEGARI, B. ; SILVA, A. A. C. E. ; GIVAGO DA SILVA, SOUZA . The role of sex and handedness in the performance of the smartphone-based Finger-Tapping Test. PLOS Digital Health, v. 2, p. e0000304, 2023.</a>.</p> 
    <a href="https://www.scielo.br/j/aabc/a/7z5HDVZKYVMxfWm8HxcJqZG/?lang=en&format=pdf" target="_blank" style="color:#1E90FF; text-decoration:none;">14. LIMA, R. C. ; BRITO, F. A. C. ; NASCIMENTO, R. L. ; MARTINS, S. N. E. S. ; MONTEIRO, L. C. P. ; SEABRA, J. P. ; FARIA, H. L. C. ; SILVA, L. M. C. ; MIRANDA, V. M. S. ; BELGAMO, A. ; ANDRÉ DOS SANTOS, CABRAL ; CALLEGARI, B. ; COSTA E SILVA, A. A ; CRISP, A. ; ALVES, CÂNDIDA HELENA LOPES ; LACERDA, E. M. C. B. ; SOUZA, G.S. . DATASET OF SMARTPHONE-BASED FINGER TAPPING TEST. Scientific Data, v. 1, p. 1, 2024.</a>.</p> 
    <a href="https://www.scielo.br/j/aabc/a/7z5HDVZKYVMxfWm8HxcJqZG/?lang=en&format=pdf" target="_blank" style="color:#1E90FF; text-decoration:none;">15. ALMEIDA, J. R. ; MONTEIRO, L. C. P. ; SOUZA, P. H. C. ; ANDRÉ DOS SANTOS, CABRAL ; BELGAMO, A. ; COSTA E SILVA, A. A ; CRISP, A. ; CALLEGARI, B. ; AVILA, P. E. S. ; SILVA, J. A. ; BASTOS, G. N. T. ; SOUZA, G.S. . Comparison of joint position sense measured by inertial sensors embedded in portable digital devices with different masses. Frontiers in Neuroscience, v. 19, p. 1-1, 2025.</a>.</p> 
    </p> </div> """)
    st.markdown(html, unsafe_allow_html=True)






























































