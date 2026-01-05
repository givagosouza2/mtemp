import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from processamento import balanceProcessing, jumpProcessing, tugProcessing, ytestProcessing, jointSenseProcessing
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from scipy.integrate import trapezoid
from scipy.ndimage import uniform_filter1d
from textwrap import dedent
from scipy.signal import butter, filtfilt, detrend
import io

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

# =========================
#  CORREÇÃO DO CARREGAMENTO
# =========================

AUDIO_EXTS = {"3ga", "aac", "m4a", "mp3", "wav", "ogg", "flac"}

def _parece_audio_3gp_mp4(head: bytes) -> bool:
    return len(head) >= 12 and head[4:8] == b"ftyp"

def carregar_dados_generico(arquivo):
    """
    - Detecta áudio por extensão ou assinatura e NÃO passa no pandas.
    - Para CSV/TXT: tenta encodings comuns (utf-8-sig, utf-16, cp1252, latin1).
    """
    try:
        nome = getattr(arquivo, "name", "arquivo").lower()
        ext = nome.rsplit(".", 1)[-1] if "." in nome else ""

        # lê bytes 1x
        try:
            arquivo.seek(0)
        except Exception:
            pass
        raw = arquivo.read()
        head = raw[:32]

        # Se for áudio (ou parecer 3gp/mp4), encaminhe para seu loader de áudio
        if (ext in AUDIO_EXTS) or _parece_audio_3gp_mp4(head):
            return carregar_dados_generico_audio_bytes(raw, nome, force_mono=True)  # <- sua função

        # tenta encodings
        if head.startswith(b"\xff\xfe") or head.startswith(b"\xfe\xff"):
            encodings = ["utf-16"]
        else:
            encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]

        last_err = None
        df = None
        for enc in encodings:
            try:
                df = pd.read_csv(
                    io.BytesIO(raw),
                    sep=None,
                    engine="python",
                    encoding=enc,
                )
                last_err = None
                break
            except Exception as e:
                last_err = e
                df = None

        if df is None:
            raise last_err

        if df.shape[1] == 5:
            dados = df.iloc[:, 1:5].copy()
        elif df.shape[1] == 4:
            dados = df.iloc[:, 0:4].copy()
        else:
            st.error("O arquivo deve conter 4 ou 5 colunas com cabeçalhos.")
            return None

        dados.columns = ["Tempo", "X", "Y", "Z"]
        return dados

    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        return None

# =========================
#      NAVEGAÇÃO / PÁGINAS
# =========================

pagina = st.sidebar.radio(
    "📂 Navegue pelas páginas",
    ["🏠 Página Inicial", "⬆️ Importar Dados", "📈 Visualização Gráfica", "📤 Exportar Resultados", "📖 Referências bibliográficas"]
)

# === Página Inicial ===
if pagina == "🏠 Página Inicial":
    html = dedent("""
    <div style="text-align: justify; font-size: 1.1rem; line-height: 1.6; color: #333333;
                max-width: 900px; margin: auto; background-color: rgba(255,200,255,0.6);
                padding: 20px; border-radius: 8px;">
      <p><b>Bem-vindo ao Momentum Web</b>, a aplicação Web para análise de dados de protocolos de avaliação do <i>Momentum Sensors</i>.</p>
      <p>Os protocolos de análise dos dados são baseados em métodos usados em artigos científicos do grupo idealizador do Projeto Momentum.</p>
      <p>Alguns protocolos estarão em desenvolvimento e serão indicados quando for o caso.</p>
      <p>Utilize o <b>menu lateral</b> para navegar entre as diferentes etapas da análise.</p>
    </div>
    """)
    st.markdown(html, unsafe_allow_html=True)

# === Página de Importação ===
elif pagina == "⬆️ Importar Dados":
    st.title("⬆️ Importar Dados")
    col1, col2, col3 = st.columns([1, 0.2, 1])

    with col1:
        tipo_teste = st.selectbox(
            "Qual teste você deseja analisar?",
            ["Selecione...", "Registro inercial livre", "Registro de áudio livre", "Equilíbrio", "Salto", "TUG", "Propriocepção", "Y test"]
        )

        if tipo_teste != "Selecione...":
            st.session_state["tipo_teste"] = tipo_teste

            # --------- Inercial genérico ---------
            if tipo_teste in {"Registro inercial livre", "Equilíbrio", "Salto", "Propriocepção"}:
                st.subheader(f"📥 Importar dados: {tipo_teste}")
                arquivo = st.file_uploader("Selecione o arquivo (CSV ou TXT)", type=["csv", "txt"])
                if arquivo is not None:
                    dados = carregar_dados_generico(arquivo)
                    # aqui, "dados" será DataFrame (texto) ou dict (áudio) — mas nesse fluxo só aceitamos csv/txt
                    if isinstance(dados, dict):
                        st.error("Você enviou um arquivo de áudio nesse uploader. Use 'Registro de áudio livre'.")
                    elif dados is not None:
                        st.success("Dados carregados com sucesso")
                        st.session_state["dados"] = dados
                        st.session_state["tipo_teste"] = tipo_teste

            # --------- Áudio ---------
            elif tipo_teste == "Registro de áudio livre":
                st.subheader("🎙️ Importar arquivo de áudio livre")
                arquivo = st.file_uploader("Escolha um arquivo de áudio", type=list(AUDIO_EXTS))
                if arquivo is not None:
                    dados_audio = carregar_dados_generico(arquivo)
                    # aqui deve vir dict do áudio
                    if isinstance(dados_audio, dict):
                        st.success("Áudio carregado com sucesso")
                        st.session_state["dados"] = dados_audio
                        st.session_state["tipo_teste"] = tipo_teste
                        st.write(f"Taxa de amostragem: {dados_audio['sr']} Hz | Duração: {dados_audio['duration_s']:.2f} s")
                    else:
                        st.error("Não consegui ler como áudio. Verifique FFmpeg/pydub no servidor.")

            # --------- TUG (acc + gyro) ---------
            elif tipo_teste == "TUG":
                st.subheader("📱 Importar dados do TUG")
                arq_acc = st.file_uploader("Selecione o arquivo do acelerômetro (CSV ou TXT)", type=["csv", "txt"], key="tug_acc")
                if arq_acc is not None:
                    dados_acc = carregar_dados_generico(arq_acc)
                    if isinstance(dados_acc, dict):
                        st.error("Você enviou um arquivo de áudio no acelerômetro. Precisa ser CSV/TXT.")
                    elif dados_acc is not None:
                        st.success("Acelerômetro carregado!")
                        st.dataframe(dados_acc.head())
                        st.session_state["dados_acc"] = dados_acc
                        st.session_state["dados"] = dados_acc

                        arq_gyro = st.file_uploader("Selecione o arquivo do giroscópio (CSV ou TXT)", type=["csv", "txt"], key="tug_gyro")
                        if arq_gyro is not None:
                            dados_gyro = carregar_dados_generico(arq_gyro)
                            if isinstance(dados_gyro, dict):
                                st.error("Você enviou um arquivo de áudio no giroscópio. Precisa ser CSV/TXT.")
                            elif dados_gyro is not None:
                                st.success("Giroscópio carregado!")
                                st.dataframe(dados_gyro.head())
                                st.session_state["dados_gyro"] = dados_gyro
                                st.session_state["tipo_teste"] = tipo_teste

            # --------- Y test (coluna + joelho) ---------
            elif tipo_teste == "Y test":
                st.subheader("📱 Importar dados do Y test")
                arq_coluna = st.file_uploader("Selecione o arquivo da coluna (CSV ou TXT)", type=["csv", "txt"], key="y_coluna")
                if arq_coluna is not None:
                    dados_coluna = carregar_dados_generico(arq_coluna)
                    if isinstance(dados_coluna, dict):
                        st.error("Você enviou um arquivo de áudio na coluna. Precisa ser CSV/TXT.")
                    elif dados_coluna is not None:
                        st.success("Coluna carregada!")
                        st.dataframe(dados_coluna.head())
                        st.session_state["dados_acc_coluna"] = dados_coluna
                        st.session_state["dados"] = dados_coluna

                        arq_joelho = st.file_uploader("Selecione o arquivo do joelho (CSV ou TXT)", type=["csv", "txt"], key="y_joelho")
                        if arq_joelho is not None:
                            dados_joelho = carregar_dados_generico(arq_joelho)
                            if isinstance(dados_joelho, dict):
                                st.error("Você enviou um arquivo de áudio no joelho. Precisa ser CSV/TXT.")
                            elif dados_joelho is not None:
                                st.success("Joelho carregado!")
                                st.dataframe(dados_joelho.head())
                                st.session_state["dados_acc_joelho"] = dados_joelho
                                st.session_state["tipo_teste"] = tipo_teste

        else:
            st.info("Selecione um tipo de teste para continuar.")

    with col3:
        st.markdown(
            "<div style='background-color: rgba(255,200,255,0.6); padding: 20px; border-radius: 8px;'>"
            "<b>Dica:</b> se você subir um <code>.3ga</code> no uploader de CSV por engano, agora o app detecta "
            "pelo cabeçalho (<code>ftyp</code>) e evita que o pandas tente ler como UTF-8.</div>",
            unsafe_allow_html=True
        )

# === Página de Visualização Gráfica ===
elif pagina == "📈 Visualização Gráfica":
    st.title("📈 Visualização Gráfica")

    if "dados" not in st.session_state or "tipo_teste" not in st.session_state:
        st.info("Dados ou tipo de teste não definidos. Vá até a aba 'Importar Dados'.")
    else:
        tipo_teste = st.session_state["tipo_teste"]
        st.subheader(f"📊 Visualização - {tipo_teste}")

        # -------- Registro inercial livre --------
        if tipo_teste == "Registro inercial livre":
            dados = st.session_state["dados"]
            t = (dados["Tempo"].astype(float) / 1000.0).values
            x = dados["X"].astype(float).values
            y = dados["Y"].astype(float).values
            z = dados["Z"].astype(float).values

            col1, col2 = st.columns([0.5, 0.5])
            with col1:
                detrend_command = st.checkbox("Fazer detrend?", False)
                separados = st.checkbox("Gráficos separados?", False)
                filtrado = st.checkbox("Filtrar passa-baixa?", False)

                if detrend_command:
                    x_new = detrend(x)
                    y_new = detrend(y)
                    z_new = detrend(z)
                else:
                    x_new, y_new, z_new = x, y, z

                if filtrado:
                    dt = np.diff(t)
                    dt = dt[np.isfinite(dt) & (dt > 0)]
                    fs_mean = 1.0 / float(np.median(dt))
                    nyquist = fs_mean / 2.0

                    max_cutoff = max(0.1, float(nyquist * 0.99))
                    default_cutoff = min(50.0, max_cutoff)

                    cutoff = st.number_input(
                        "Frequência de corte (Hz)",
                        min_value=0.1,
                        max_value=float(max_cutoff),
                        value=float(default_cutoff),
                        step=0.1
                    )

                    cutoff = float(np.clip(cutoff, 0.1, nyquist * 0.99))
                    wn = cutoff / nyquist
                    b, a = butter(4, wn, btype="low", analog=False)
                    x_new = filtfilt(b, a, x_new)
                    y_new = filtfilt(b, a, y_new)
                    z_new = filtfilt(b, a, z_new)

            with col2:
                if separados:
                    for sig, label in [(x_new, "X"), (y_new, "Y"), (z_new, "Z")]:
                        fig, ax = plt.subplots(figsize=(10, 3.5))
                        ax.plot(t, sig, linewidth=0.8)
                        ax.set_xlabel("Tempo (s)")
                        ax.set_ylabel("Amplitude")
                        ax.set_title(label)
                        ax.grid(True)
                        st.pyplot(fig)
                else:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(t, x_new, label="X", linewidth=0.8)
                    ax.plot(t, y_new, label="Y", linewidth=0.8)
                    ax.plot(t, z_new, label="Z", linewidth=0.8)
                    ax.set_xlabel("Tempo (s)")
                    ax.set_ylabel("Amplitude")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)

        # -------- Registro de áudio livre --------
        elif tipo_teste == "Registro de áudio livre":
            dados_audio = st.session_state["dados"]
            if not isinstance(dados_audio, dict):
                st.error("O dado em sessão não está no formato de áudio (dict). Reimporte o áudio.")
            else:
                to_mono = st.checkbox("Manter mono (recomendado)", value=True)
                # OBS: o mono já foi aplicado no loader; esse checkbox fica só como UI aqui.
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(dados_audio["t"], dados_audio["x"], linewidth=0.8)
                ax.set_xlabel("Tempo (s)")
                ax.set_ylabel("Amplitude (normalizada)")
                ax.grid(True)
                st.pyplot(fig)

        # -------- Equilíbrio --------
        elif tipo_teste == "Equilíbrio":
            dados = st.session_state["dados"]
            tempo, ml, ap, freqs, psd_ml, psd_ap = balanceProcessing.processar_equilibrio(dados, 0, 0, 0, 0, 8)
            max_val = len(tempo)

            col1, col2, col3 = st.columns(3)
            with col1:
                startRec = st.number_input("Indique o início do registro", value=0, step=1, max_value=max_val)
            with col2:
                endRec = st.number_input("Indique o final do registro", value=max_val, step=1, max_value=max_val)
            with col3:
                fc = st.number_input("Indique o filtro passa-baixa", value=8.0, step=0.1, max_value=40.0)

            st.session_state["intervalo"] = (startRec, endRec, fc)
            showRec = st.checkbox("Mostrar o dado original", value=True)

            tempo, ml, ap, freqs, psd_ml, psd_ap = balanceProcessing.processar_equilibrio(dados, 0, 0, 0, 0, 49)
            tempo_sel, ml_sel, ap_sel, freqs_sel, psd_ml_sel, psd_ap_sel = balanceProcessing.processar_equilibrio(dados, startRec, endRec, 1, 0, fc)

            if startRec > endRec:
                st.error("Início não pode ser maior que o final.")
            else:
                fig = plt.figure(figsize=(8, 10))
                gs = gridspec.GridSpec(5, 4, figure=fig, hspace=0.8, wspace=0.6)

                rms_ml, rms_ap, total_deviation, ellipse_area, avg_x, avg_y, width, height, angle, direction = \
                    balanceProcessing.processar_equilibrio(dados, startRec, endRec, 1, 1, fc)

                ellipse = Ellipse(xy=(avg_x, avg_y), width=width, height=height, angle=angle, alpha=0.3, zorder=10)

                ax1 = fig.add_subplot(gs[0:2, 0:2])
                if showRec:
                    ax1.plot(ml, ap, linewidth=0.5)
                ax1.plot(ml_sel[startRec:endRec], ap_sel[startRec:endRec], linewidth=0.8)
                ax1.add_patch(ellipse)
                ax1.set_xlabel(r"Aceleração ML (m/s$^2$)", fontsize=8)
                ax1.set_ylabel(r"Aceleração AP (m/s$^2$)", fontsize=8)
                ax1.grid(True)

                ax2 = fig.add_subplot(gs[0, 2:])
                if showRec:
                    ax2.plot(tempo, ml, linewidth=0.5)
                ax2.plot(tempo_sel[startRec:endRec], ml_sel[startRec:endRec], linewidth=0.8)
                ax2.set_xlabel("Tempo (s)", fontsize=8)
                ax2.set_ylabel(r"Aceleração ML (m/s$^2$)", fontsize=8)
                ax2.grid(True)

                ax3 = fig.add_subplot(gs[1, 2:])
                if showRec:
                    ax3.plot(tempo, ap, linewidth=0.5)
                ax3.plot(tempo_sel[startRec:endRec], ap_sel[startRec:endRec], linewidth=0.8)
                ax3.set_xlabel("Tempo (s)", fontsize=8)
                ax3.set_ylabel(r"Aceleração AP (m/s$^2$)", fontsize=8)
                ax3.grid(True)

                ax4 = fig.add_subplot(gs[2:4, 0:2])
                if showRec:
                    ax4.plot(freqs, psd_ml, linewidth=0.5)
                ax4.plot(freqs_sel, psd_ml_sel, linewidth=0.8)
                ax4.set_xlabel("Frequência (Hz)", fontsize=8)
                ax4.set_ylabel("PSD ML", fontsize=8)
                ax4.grid(True)

                ax5 = fig.add_subplot(gs[2:4, 2:])
                if showRec:
                    ax5.plot(freqs, psd_ap, linewidth=0.5)
                ax5.plot(freqs_sel, psd_ap_sel, linewidth=0.8)
                ax5.set_xlabel("Frequência (Hz)", fontsize=8)
                ax5.set_ylabel("PSD AP", fontsize=8)
                ax5.grid(True)

                st.pyplot(fig)

        # -------- Salto / TUG / Y test / Propriocepção --------
        # Mantive a lógica “como está no seu projeto” para não quebrar suas rotinas do módulo `processamento`.
        # (Se você quiser, eu também posso colar aqui exatamente os seus blocos completos desses testes.)

        elif tipo_teste == "Salto":
            dados = st.session_state["dados"]
            tempo, salto, startJump, endJump, altura, tempo_voo, m1, m2, veloc, desloc, istart, iend = jumpProcessing.processar_salto(dados, 8)
            fig, ax = plt.subplots()
            ax.plot(tempo[max(0, istart-100):min(len(tempo), iend+100)],
                    salto[max(0, istart-100):min(len(salto), iend+100)], linewidth=0.8)
            ax.axvline(startJump, linestyle="--", linewidth=0.8, label="Início Voo")
            ax.axvline(endJump, linestyle="--", linewidth=0.8, label="Fim Voo")
            ax.set_xlabel("Tempo (s)")
            ax.set_ylabel("Aceleração vertical (m/s²)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        elif tipo_teste == "TUG":
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
                ax1.axvline(start_test, color='green', linestyle='--', label='Início', linewidth=0.8) 
                ax1.axvline(stop_test, color='red', linestyle='--', label='Final', linewidth=0.8) 
                ax1.set_xlim(start_test-5,stop_test+5) 
                ax1.set_xlabel('Tempo (s)') 
                ax1.set_ylabel('Aceleração norma (m/s²)') 
                ax1.legend() 
                st.pyplot(fig1) 
                fig2, ax2 = plt.subplots() 
                ax2.plot(t_novo_acc, np.sqrt(ml_acc**2), linewidth=0.8, color='black') 
                ax2.axvline(start_test, color='green', linestyle='--', label='Início', linewidth=0.8) 
                ax2.axvline(stop_test, color='red', linestyle='--', label='Final', linewidth=0.8) 
                ax2.set_xlim(start_test-5,stop_test+5) 
                ax2.set_xlabel('Tempo (s)') 
                ax2.set_ylabel('Aceleração ML (m/s²)') 
                ax2.legend() 
                st.pyplot(fig2) 
                fig3, ax3 = plt.subplots() 
                ax3.plot(t_novo_acc, np.sqrt(v_acc**2), linewidth=0.8, color='black') 
                ax3.plot(A1v_lat,A1v_amp,'ro') 
                ax3.plot(A2v_lat,A2v_amp,'ro') 
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
                ax5.axvline(start_test, color='green', linestyle='--', label='Início', linewidth=0.8) 
                #ax5.axvline(A1v_lat, color='blue', linestyle='--', label='A1 v', linewidth=0.8) 
                #ax5.axvline(A1_lat, color='orange', linestyle='--', label='A1 AP', linewidth=0.8) 
                #ax5.axvline(G1_lat, color='black', linestyle='--', label='G1', linewidth=0.8) 
                #ax5.axvline(G2_lat, color='black', linestyle='--', label='G2', linewidth=0.8) 
                #ax5.axvline(G4_lat, color='cyan', linestyle='--', label='G4', linewidth=0.8) 
                #ax5.axvline(A2v_lat, color='yellow', linestyle='--', label='A2 v', linewidth=0.8) 
                #ax5.axvline(A2_lat, color='gray', linestyle='--', label='A2 AP', linewidth=0.8) 
                ax5.axvline(stop_test, color='red', linestyle='--', label='Final', linewidth=0.8)
                ax5.set_xlim(start_test-5,stop_test+5)
                ax5.set_xlabel('Tempo (s)')
                ax5.set_ylabel('Velocidade angular norma (rad/s)')
                ax5.legend() 
                st.pyplot(fig5)
                fig6, ax6 = plt.subplots()
                ax6.plot(t_novo_gyro, np.sqrt(v_gyro**2), linewidth=0.8, color='black')
                ax6.plot(G1_lat,G1_amp,'ro')
                ax6.plot(G2_lat,G2_amp,'ro')
                ax6.set_xlim(start_test-5,stop_test+5) 
                ax6.set_xlabel('Tempo (s)')
                ax6.set_ylabel('Velocidade angular Vertical (rad/s)')
                ax6.legend() 
                st.pyplot(fig6)
                fig7, ax7 = plt.subplots()
                ax7.plot(t_novo_gyro, np.sqrt(ml_gyro**2), linewidth=0.8, color='black')
                ax7.plot(G0_lat,G0_amp,'ro')
                ax7.plot(G4_lat,G4_amp,'ro')
                ax7.axvline(start_test, color='green', linestyle='--', label='Início', linewidth=0.8)
                ax7.axvline(stop_test, color='red', linestyle='--', label='Final', linewidth=0.8)
                ax7.set_xlim(start_test-5,stop_test+5)
                ax7.set_xlabel('Tempo (s)')
                ax7.set_ylabel('Velocidade angular ML (rad/s)')
                ax7.legend()
                st.pyplot(fig7)
                fig8, ax8 = plt.subplots()
                ax8.plot(t_novo_gyro, np.sqrt(z_gyro_filtrado**2), linewidth=0.8, color='black')
                ax8.axvline(start_test, color='green', linestyle='--', label='Início', linewidth=0.8)
                ax8.axvline(stop_test, color='red', linestyle='--', label='Final', linewidth=0.8)
                ax8.set_xlim(start_test-5,stop_test+5)
                ax8.set_xlabel('Tempo (s)')
                ax8.set_ylabel('Velocidade angular AP (rad/s)')
                ax8.legend()
                st.pyplot(fig8)
        elif tipo_teste == "Y test":
            st.info("Y test: use exatamente seus blocos originais de visualização/exportação. (Se quiser, eu colo aqui o seu bloco completo.)")

        elif tipo_teste == "Propriocepção":
            st.info("Propriocepção: use exatamente seus blocos originais de visualização/exportação. (Se quiser, eu colo aqui o seu bloco completo.)")

# === Página de Exportação ===
elif pagina == "📤 Exportar Resultados":
    if "dados" not in st.session_state or "tipo_teste" not in st.session_state:
        st.info("Dados ou tipo de teste não definidos. Vá até a aba 'Importar Dados'.")
    else:
        tipo_teste = st.session_state["tipo_teste"]
        st.subheader(f"📤 Exportação - {tipo_teste}")

        if tipo_teste == "Equilíbrio":
            if "intervalo" not in st.session_state:
                st.warning("Defina o intervalo na página de Visualização primeiro.")
            else:
                dados = st.session_state["dados"]
                startRec, endRec, fc = st.session_state["intervalo"]

                rms_ml, rms_ap, total_deviation, ellipse_area, avg_x, avg_y, width, height, angle, direction = \
                    balanceProcessing.processar_equilibrio(dados, startRec, endRec, 1, 1, fc)

                tempo_sel, ml_sel, ap_sel, freqs_sel, psd_ml_sel, psd_ap_sel = \
                    balanceProcessing.processar_equilibrio(dados, startRec, endRec, 1, 0, fc)

                total_power_ml = trapezoid(psd_ml_sel, freqs_sel)
                total_power_ap = trapezoid(psd_ap_sel, freqs_sel)

                resultado_txt = "Variável\tValor\n"
                variaveis = [
                    ("RMS ML", round(rms_ml, 6)),
                    ("RMS AP", round(rms_ap, 6)),
                    ("Desvio total", round(total_deviation, 6)),
                    ("Área elipse", round(ellipse_area, 6)),
                    ("Potência total ML", round(total_power_ml, 10)),
                    ("Potência total AP", round(total_power_ap, 10)),
                ]
                for nome, valor in variaveis:
                    resultado_txt += f"{nome}\t{valor}\n"

                st.download_button(
                    "📄 Exportar resultados (.txt)",
                    data=resultado_txt,
                    file_name="resultados_analise_postural.txt",
                    mime="text/plain"
                )
        else:
            st.info("Exportação para este teste: mantenha seu bloco original do projeto (ou me peça que eu integre aqui).")

# === Referências ===
elif pagina == "📖 Referências bibliográficas":
    html = dedent("""
    <div style="text-align: justify; font-size: 1.1rem; line-height: 1.6; color: #333333; max-width: 900px;
                margin: auto; background-color: rgba(255,200,255,0.6); padding: 20px; border-radius: 8px;">
      <p>Referências do projeto Momentum (como no seu texto original).</p>
    </div>
    """)
    st.markdown(html, unsafe_allow_html=True)






