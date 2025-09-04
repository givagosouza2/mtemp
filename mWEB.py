import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from processamento import balanceProcessing
from processamento import jumpProcessing
from processamento import tugProcessing
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from scipy.integrate import trapezoid, cumulative_trapezoid

st.set_page_config(page_title="App teste", layout="wide")

# Função genérica para carregar dados de arquivos com 4 ou 5 colunas


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


pagina = st.sidebar.radio("📂 Navegue pelas páginas", [
    "🏠 Página Inicial",
    "📁 Importar Dados",
    "📈 Visualização Gráfica",
    "📤 Exportar Resultados"
])

# === Página Inicial ===
if pagina == "🏠 Página Inicial":
    st.title("👋 Bem-vindo ao App de Análise de Dados Digitais")
    st.write(
        "Utilize o menu lateral para navegar entre as diferentes etapas da análise.")

# === Página de Importação ===
elif pagina == "📁 Importar Dados":
    st.title("📁 Importar Dados")

    col1, col2, col3 = st.columns([1, 0.2, 1])
    with col1:
        tipo_teste = st.selectbox(
            "Qual teste você deseja analisar?",
            ["Selecione...", "Equilíbrio", "Salto", "TUG"]
        )

        if tipo_teste != "Selecione...":
            st.session_state["tipo_teste"] = tipo_teste

        if tipo_teste == "Equilíbrio":
            st.subheader("📦 Importar dados de Equilíbrio")
            arquivo = st.file_uploader(
                "Selecione o arquivo de equilíbrio (CSV ou TXT)", type=["csv", "txt"])
            if arquivo is not None:
                dados = carregar_dados_generico(arquivo)
                if dados is not None:
                    st.success('Dados carregados com sucesso')
                    st.session_state["dados"] = dados

        elif tipo_teste == "Salto":
            st.subheader("👆 Importar dados de Salto")
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

        elif tipo_teste == "Selecione...":
            st.info("Selecione um tipo de teste para continuar.")
    with col3:
        if tipo_teste == "Equilíbrio":
            st.title('Equilíbrio estático. ')
        elif tipo_teste == 'Salto':
            st.title('Salto')
        else:
            st.title('Men at working')


# === Página de Visualização Gráfica ===
elif pagina == "📈 Visualização Gráfica":
    st.title("📈 Visualização Gráfica")
    if "dados" in st.session_state and "tipo_teste" in st.session_state:
        tipo_teste = st.session_state["tipo_teste"]
        st.subheader(f"📊 Visualização - {tipo_teste}")
        #######
        if tipo_teste == "Equilíbrio":
            dados = st.session_state["dados"]
            tempo, ml, ap, freqs, psd_ml, psd_ap = balanceProcessing.processar_equilibrio(
                dados, 0, 0, 0, 0, 8)
            max_val = len(tempo)

            col1, col2, col3 = st.columns(3)
            with col1:
                startRec = st.number_input(
                    'Indique o início do registro', value=0, step=1, max_value=max_val)
            with col2:
                endRec = st.number_input(
                    'Indique o final do registro', value=max_val, step=1, max_value=max_val)
            with col3:
                filter = st.number_input(
                    'Indique o filtro passa-baixa', value=8.0, step=0.1, max_value=40.0)
            st.session_state["intervalo"] = startRec, endRec, filter
            showRec = st.checkbox('Mostrar o dado original', value=True)
            tempo, ml, ap, freqs, psd_ml, psd_ap = balanceProcessing.processar_equilibrio(
                dados, 0, 0, 0, 0, 49)
            tempo_sel, ml_sel, ap_sel, freqs_sel, psd_ml_sel, psd_ap_sel = balanceProcessing.processar_equilibrio(
                dados, startRec, endRec, 1, 0, filter)
            if startRec > endRec:
                st.error(
                    'Valor do início do registro não pode ser maior que o do final do registro')
            else:
                if endRec > max_val:
                    st.error(
                        'Valor do início do registro não pode ser maior que o do final do registro')
                else:
                    min_ml = np.min(ml)
                    max_ml = np.max(ml)
                    min_ap = np.min(ap)
                    max_ap = np.max(ap)
                    limite = max(np.abs(min_ml), np.abs(
                        max_ml), np.abs(min_ap), np.abs(max_ap))
                    if limite < 0.5:
                        limite = 0.5
                    elif limite >= 0.5 and limite < 5:
                        limite = 5
                    elif limite >= 5 and limite < 10:
                        limite = 10
                    else:
                        limite = 50

                    # Cria figura com GridSpec personalizado
                    fig = plt.figure(figsize=(8, 10))
                    gs = gridspec.GridSpec(
                        5, 4, figure=fig, hspace=0.8, wspace=0.6)

                    # Gráfico 1: ocupa 2x2 blocos (esquerda acima)

                    rms_ml, rms_ap, total_deviation, ellipse_area, avg_x, avg_y, width, height, angle, direction = balanceProcessing.processar_equilibrio(
                        dados, startRec, endRec, 1, 1, filter)

                    ellipse = Ellipse(xy=(avg_x, avg_y), width=width, height=height,
                                      angle=angle, alpha=0.5, color='blue', zorder=10)
                    ax1 = fig.add_subplot(gs[0:2, 0:2])

                    if showRec:
                        ax1.plot(ml, ap, color='tomato', linewidth=0.5)
                    ax1.plot(
                        ml_sel[startRec:endRec], ap_sel[startRec:endRec], color='black', linewidth=0.8)
                    ax1.set_xlabel(r'Aceleração ML (m/s$^2$)', fontsize=8)
                    ax1.set_ylabel(r'Aceleração AP (m/s$^2$)', fontsize=8)
                    ax1.set_xlim(-limite, limite)
                    ax1.set_ylim(-limite, limite)
                    ax1.tick_params(axis='both', labelsize=8)
                    ax1.add_patch(ellipse)

                    # Gráfico 2: ocupa linha superior direita (metade superior)
                    ax2 = fig.add_subplot(gs[0, 2:])
                    if showRec:
                        ax2.plot(tempo, ml, color='tomato', linewidth=0.5)
                    ax2.plot(
                        tempo_sel[startRec:endRec], ml_sel[startRec:endRec], color='black', linewidth=0.8)
                    ax2.set_xlabel('Tempo (s)', fontsize=8)
                    ax2.set_ylabel(r'Aceleração ML (m/s$^2$)', fontsize=8)
                    ax2.set_xlim(0, max(tempo))
                    ax2.set_ylim(-limite, limite)
                    ax2.tick_params(axis='both', labelsize=8)

                    # Gráfico 3: linha do meio à direita
                    ax3 = fig.add_subplot(gs[1, 2:])
                    if showRec:
                        ax3.plot(tempo, ap, color='tomato', linewidth=0.5)
                    ax3.plot(
                        tempo_sel[startRec:endRec], ap_sel[startRec:endRec], color='black', linewidth=0.8)
                    ax3.set_xlabel('Tempo (s)', fontsize=8)
                    ax3.set_ylabel(r'Aceleração AP (m/s$^2$)', fontsize=8)
                    ax3.set_xlim(0, max(tempo))
                    ax3.set_ylim(-limite, limite)
                    ax3.tick_params(axis='both', labelsize=8)

                    # Gráfico 4: canto inferior esquerdo
                    ax4 = fig.add_subplot(gs[2:4, 0:2])
                    if showRec:
                        ax4.plot(freqs, psd_ml, color='tomato', linewidth=0.5)
                    ax4.plot(freqs_sel, psd_ml_sel, 'k')
                    ax4.set_xlim(-0.1, 8)
                    ax4.set_ylim(0, limite*0.025)
                    ax4.set_xlabel('Frequência temporal (Hz)', fontsize=8)
                    ax4.set_ylabel(r'Aceleração ML (m/s$^2$)', fontsize=8)
                    ax4.tick_params(axis='both', labelsize=8)

                    # Gráfico 5: canto inferior direito
                    ax5 = fig.add_subplot(gs[2:4, 2:])
                    if showRec:
                        ax5.plot(freqs, psd_ap, color='tomato', linewidth=0.5)
                    ax5.plot(freqs_sel, psd_ap_sel, 'k')
                    ax5.set_xlim(-0.1, 8)
                    ax5.set_ylim(0, limite*0.025)
                    ax5.set_xlabel('Frequência temporal (Hz)', fontsize=8)
                    ax5.set_ylabel(r'Aceleração AP (m/s$^2$)', fontsize=8)
                    ax5.tick_params(axis='both', labelsize=8)

                    # Exibe no Streamlit
                    st.pyplot(fig)

        if tipo_teste == "Salto":
            col1, col2, col3 = st.columns([0.4, 1, 0.4])
            dados = st.session_state["dados"]
            tempo, salto, startJump, endJump, altura, tempo_voo, m1, m2, veloc, desloc, istart, iend = jumpProcessing.processar_salto(
                dados, 8)
            with col2:
                fig, ax = plt.subplots()
                ax.plot(tempo[istart-100:iend+100], salto[istart -
                        100:iend+100], linewidth=0.8, color='black')
                ax.axvline(startJump, color='green',
                           linestyle='--', label='Início Voo', linewidth=0.8)
                ax.axvline(endJump, color='red',
                           linestyle='--', label='Fim Voo', linewidth=0.8)
                ax.set_xlabel('Tempo (s)')
                ax.set_ylabel('Aceleração vertical (m/s²)')
                ax.legend()
                st.pyplot(fig)
        
        if tipo_teste == "TUG":
            col1, col2, col3 = st.columns([0.4, 0.4, 0.4])
            dados_acc = st.session_state["dados_acc"]
            dados_gyro = st.session_state["dados_gyro"]
            t_novo_acc, x_acc_filtrado, y_acc_filtrado, z_acc_filtrado, norma_acc_filtrado, t_novo_gyro, v_gyro, ml_gyro, z_gyro_filtrado, norma_gyro_filtrado,start_test,stop_test,idx,idx_ml,duration = tugProcessing.processar_tug(dados_acc,dados_gyro,4,1.25)
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
            amp1 = vertical_squared[idx_ml[0][0]]
            amp2 = vertical_squared[idx_ml[0][1]]
            
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
            
                
            with col1:
                fig1, ax1 = plt.subplots()
                ax1.plot(t_novo_acc, norma_acc_filtrado, linewidth=0.8, color='black')
                ax1.axvline(start_test, color='green',
                           linestyle='--', label='Início', linewidth=0.8)
                ax1.axvline(stop_test, color='red',
                           linestyle='--', label='Final', linewidth=0.8)
                ax1.set_xlabel('Tempo (s)')
                ax1.set_ylabel('Aceleração vertical (m/s²)')
                ax1.legend()
                st.pyplot(fig1)

                fig2, ax2 = plt.subplots()
                ax2.plot(t_novo_acc, np.sqrt(x_acc_filtrado**2), linewidth=0.8, color='black')
                ax2.axvline(start_test, color='green',
                           linestyle='--', label='Início', linewidth=0.8)
                ax2.axvline(stop_test, color='red',
                           linestyle='--', label='Final', linewidth=0.8)
                ax2.set_xlabel('Tempo (s)')
                ax2.set_ylabel('Aceleração vertical (m/s²)')
                ax2.legend()
                st.pyplot(fig2)

                fig3, ax3 = plt.subplots()
                ax3.plot(t_novo_acc, np.sqrt(y_acc_filtrado**2), linewidth=0.8, color='black')
                ax3.axvline(start_test, color='green',
                           linestyle='--', label='Início', linewidth=0.8)
                ax3.axvline(stop_test, color='red',
                           linestyle='--', label='Final', linewidth=0.8)
                ax3.set_xlabel('Tempo (s)')
                ax3.set_ylabel('Aceleração vertical (m/s²)')
                ax3.legend()
                st.pyplot(fig3)

                fig4, ax4 = plt.subplots()
                ax4.plot(t_novo_acc, np.sqrt(z_acc_filtrado**2), linewidth=0.8, color='black')
                ax4.axvline(start_test, color='green',
                           linestyle='--', label='Início', linewidth=0.8)
                ax4.axvline(stop_test, color='red',
                           linestyle='--', label='Final', linewidth=0.8)
                ax4.set_xlabel('Tempo (s)')
                ax4.set_ylabel('Aceleração vertical (m/s²)')
                ax4.legend()
                st.pyplot(fig4)
                                    
            with col2:
                fig5, ax5 = plt.subplots()
                ax5.plot(t_novo_gyro, norma_gyro_filtrado, linewidth=0.8, color='black')
                ax5.axvline(start_test, color='green',
                           linestyle='--', label='Início', linewidth=0.8)
                ax5.axvline(stop_test, color='red',
                           linestyle='--', label='Final', linewidth=0.8)
                ax5.set_xlabel('Tempo (s)')
                ax5.set_ylabel('Velocidade angular (rad/s)')
                ax5.legend()
                st.pyplot(fig5)
                
                fig6, ax6 = plt.subplots()
                ax6.plot(t_novo_gyro, np.sqrt(v_gyro**2), linewidth=0.8, color='black')
                ax6.plot(G1_lat,G1_amp,'ro')
                ax6.plot(G2_lat,G2_amp,'ro')
                ax6.axvline(start_test, color='green',
                           linestyle='--', label='Início', linewidth=0.8)
                ax6.axvline(stop_test, color='red',
                           linestyle='--', label='Final', linewidth=0.8)
                ax6.set_xlabel('Tempo (s)')
                ax6.set_ylabel('Velocidade angular Vertical (rad/s)')
                ax6.legend()
                st.pyplot(fig6)

                fig7, ax7 = plt.subplots()
                ax7.plot(t_novo_gyro, np.sqrt(ml_gyro**2), linewidth=0.8, color='black')
                ax7.plot(G0_lat,G0_amp,'ro')
                ax7.plot(G4_lat,G4_amp,'ro')
                ax7.axvline(start_test, color='green',
                           linestyle='--', label='Início', linewidth=0.8)
                ax7.axvline(stop_test, color='red',
                           linestyle='--', label='Final', linewidth=0.8)
                ax7.set_xlabel('Tempo (s)')
                ax7.set_ylabel('Velocidade angular ML (rad/s)')
                ax7.legend()
                st.pyplot(fig7)

                fig8, ax8 = plt.subplots()
                ax8.plot(t_novo_gyro, np.sqrt(z_gyro_filtrado**2), linewidth=0.8, color='black')
                ax8.axvline(start_test, color='green',
                           linestyle='--', label='Início', linewidth=0.8)
                ax8.axvline(stop_test, color='red',
                           linestyle='--', label='Final', linewidth=0.8)
                ax8.set_xlabel('Tempo (s)')
                ax8.set_ylabel('Velocidade angular ML (rad/s)')
                ax8.legend()
                st.pyplot(fig8)
            with col3:
                st.text(idx)
            
            
        else:
            st.markdown("### Sinais brutos de X, Y e Z ao longo do Tempo")

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
            rms_ml, rms_ap, total_deviation, ellipse_area, avg_x, avg_y, width, height, angle, direction = balanceProcessing.processar_equilibrio(
                dados, startRec, endRec, 1, 1, filter)

            tempo_sel, ml_sel, ap_sel, freqs_sel, psd_ml_sel, psd_ap_sel = balanceProcessing.processar_equilibrio(
                dados, startRec, endRec, 1, 0, filter)

            planar_dev = np.rad2deg(
                np.sqrt(np.std(ml_sel)**2 + np.std(ap_sel)**2))
            sway_dir = np.rad2deg(np.arctan2(np.std(ap_sel), np.std(ml_sel)))

            total_power_ml = trapezoid(psd_ml_sel, freqs_sel)
            total_power_ap = trapezoid(psd_ap_sel, freqs_sel)

            bands = [(0, 0.5), (0.5, 2), (2, np.inf)]
            energy_ml = [trapezoid(psd_ml_sel[(freqs_sel >= b[0]) & (freqs_sel < b[1])], freqs_sel[(
                freqs_sel >= b[0]) & (freqs_sel < b[1])]) for b in bands]
            energy_ap = [trapezoid(psd_ap_sel[(freqs_sel >= b[0]) & (freqs_sel < b[1])], freqs_sel[(
                freqs_sel >= b[0]) & (freqs_sel < b[1])]) for b in bands]

            centroid_ml = trapezoid(
                freqs_sel * psd_ml_sel, freqs_sel) / total_power_ml
            centroid_ap = trapezoid(
                freqs_sel * psd_ap_sel, freqs_sel) / total_power_ap

            mode_ml = freqs_sel[np.argmax(psd_ml_sel)]
            mode_ap = freqs_sel[np.argmax(psd_ap_sel)]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(label=r"RMS ML (m/s$^2$)", value=round(rms_ml, 4))
                st.metric(label=r"RMS AP (m/s$^2$)", value=round(rms_ap, 4))
                st.metric(label=r"Desvio total (m/s$^2$)",
                          value=round(total_deviation, 4))
                st.metric(label=r"Área da elipse (m$^2$/s$^4$)",
                          value=round(ellipse_area, 4))
                st.metric(label="Desvio planar (graus)",
                          value=round(planar_dev, 4))
                st.metric(label="Direção da oscilação (graus)",
                          value=round(sway_dir, 4))
            with col2:
                st.metric(label="Centróide ML (Hz)",
                          value=round(centroid_ml, 4))
                st.metric(label="Centróide AP (Hz)",
                          value=round(centroid_ap, 4))
            with col3:
                st.metric(label=r"Energia ML 0-0.5 Hz (m/s$^2$)",
                          value=round((10**5)*energy_ml[0], 3))
                st.metric(label=r"Energia ML 0.5-2 Hz (m/s$^2$)",
                          value=round((10**5)*energy_ml[1], 3))
                st.metric(label=r"Energia ML 2-8 Hz (m/s$^2$)",
                          value=round((10**5)*energy_ml[2], 3))
            with col4:
                st.metric(label=r"Energia AP 0-0.5 Hz (m/s$^2$)",
                          value=round((10**5)*energy_ap[0], 3))
                st.metric(label=r"Energia AP 0.5-2 Hz (m/s$^2$)",
                          value=round((10**5)*energy_ap[1], 3))
                st.metric(label=r"Energia AP 2-8 Hz (m/s$^2$)",
                          value=round((10**5)*energy_ap[2], 3))
            resultado_txt = "Variável\tValor\n"  # Cabeçalho com tabulação

            # Lista de pares (nome, valor)
            variaveis = [
                ("RMS ML", round(rms_ml, 4)),
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
                ("Energia AP 2–8 Hz", round((10**5)*energy_ap[2], 8)),
            ]

            # Adiciona linha por linha
            for nome, valor in variaveis:
                resultado_txt += f"{nome}\t{valor}\n"

            st.download_button(
                label="📄 Exportar resultados (.txt)",
                data=resultado_txt,
                file_name="resultados_analise_postural.txt",
                mime="text/plain"
            )
        if tipo_teste == "Salto":
            dados = st.session_state["dados"]
            tempo, salto, startJump, endJump, altura, tempo_voo, m1, m2, veloc, desloc, istart, iend = jumpProcessing.processar_salto(
                dados, 8)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(label="Tempo de vôo (s)", value=round(tempo_voo, 4))
                st.metric(label="Altura (m)", value=round(altura, 4))
                
            with col2:
                st.metric(
                    label="Velocidade de decolagem (m/s)", value=round(veloc, 4))
                
        if tipo_teste == "TUG":
            dados_acc = st.session_state["dados_acc"]
            dados_gyro = st.session_state["dados_gyro"]
            







































