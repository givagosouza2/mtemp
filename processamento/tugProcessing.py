import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import butter, filtfilt, detrend, savgol_filter
from scipy.integrate import cumulative_trapezoid


def highpass_filter(signal, fc, fs, order=4):
    w = fc / (fs / 2)
    b, a = butter(order, w, btype='high')
    return filtfilt(b, a, signal)


def calcular_velocidade(acc, fs, metodo='savitzky', cutoff=5, ordem=3, janela=21):
    dt = 1 / fs
    tempo = np.arange(len(acc)) * dt

    # Filtragem da aceleração antes da integração
    if metodo == 'savitzky':
        acc_filtrado = savgol_filter(acc, janela, ordem)
    elif metodo == 'butterworth':
        b, a = butter(ordem, cutoff / (fs / 2), btype='low')
        acc_filtrado = filtfilt(b, a, acc)
    elif metodo == 'spline':
        spl = UnivariateSpline(tempo, acc, k=ordem)
        acc_filtrado = spl(tempo)
    else:
        raise ValueError("Método de filtragem desconhecido.")

    # Integração para velocidade
    velocidade = cumulative_trapezoid(acc_filtrado, dx=dt, initial=0)

    # Remover drift da velocidade
    velocidade = highpass_filter(velocidade, fc=0.1, fs=fs)

    return tempo, velocidade


def processar_tug(df, filter_cutoff):
    df_proc = df.copy()
    time_vec = df_proc["Tempo"]
    x = df_proc["X"]
    y = df_proc["Y"]
    z = df_proc["Z"]

    # Escolher o eixo com maior variabilidade
    v = x if np.std(x) > np.std(y) else y

    # Reamostragem para 100 Hz
    t_original = time_vec / 1000  # Converter para segundos
    fs_novo = 100
    dt_novo = 1 / fs_novo
    t_novo = np.arange(t_original.iloc[0], t_original.iloc[-1], dt_novo)

    # Interpolação
    interp_v = interp1d(t_original, v, kind='linear', fill_value="extrapolate")
    v_interp = interp_v(t_novo)

    # Remover tendência
    #v_detrended = detrend(v_interp)

    # Filtragem passa-baixa
    nyquist = fs_novo / 2
    normal_cutoff = filter_cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    v_filtrado = filtfilt(b, a, v_interp)

    # Detecção do tempo de voo
    limiar_vel = 0.5
    indices_voo = np.where(np.abs(v_filtrado) > limiar_vel)[0]

    altura = np.nan
    inicio_voo_time = np.nan
    fim_voo_time = np.nan
    tempo_voo = np.nan
    max_dec = np.nan
    max_takeoff = np.nan

    if len(indices_voo) >= 2:
        min_mov = np.argmin(v_filtrado)
        max_mov = np.argmax(v_filtrado)

        # Buscar início do voo (antes de min_mov)
        try:
            inicio_voo_idx = next(i for i in range(
                min_mov-1, -1, -1) if v_filtrado[i] > 9.81)
        except StopIteration:
            inicio_voo_idx = 0
        inicio_voo_time = t_novo[inicio_voo_idx]
        max_dec = np.max(v_filtrado[0:inicio_voo_idx])

        # Buscar fim do voo (antes de max_mov)
        try:
            fim_voo_idx = next(i for i in range(
                max_mov-1, -1, -1) if v_filtrado[i] < 9.81)
        except StopIteration:
            fim_voo_idx = len(v_filtrado) - 1
        fim_voo_time = t_novo[fim_voo_idx]
        max_takeoff = np.max(v_filtrado[fim_voo_idx:])

        tempo_voo = fim_voo_time - inicio_voo_time

        # Estimar altura via fórmula balística
        g = 9.81
        altura = (g * (tempo_voo / 2)**2) / 2
        velocidade_saida = (9.81*tempo_voo)/2

        # Cálculo da velocidade
        _, velocidade = calcular_velocidade(
            v_filtrado[inicio_voo_idx:fim_voo_idx], fs_novo)

        # Cálculo do deslocamento
        deslocamento = cumulative_trapezoid(velocidade, dx=dt_novo, initial=0)
        deslocamento = highpass_filter(deslocamento, fc=0.1, fs=fs_novo)

    return t_novo, v_filtrado, inicio_voo_time, fim_voo_time, altura, tempo_voo, max_dec, max_takeoff, velocidade_saida, deslocamento, inicio_voo_idx, fim_voo_idx
