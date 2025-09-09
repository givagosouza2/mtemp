import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, welch, detrend, savgol_filter
from typing import Tuple

def calculate_ellipse(ml_acc: np.ndarray, ap_acc: np.ndarray, confidence: float = 0.95) -> Tuple:
    cov = np.cov(ml_acc, ap_acc)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    scale = np.sqrt(vals * 5.991)  # CHI2_95
    principal_dir = np.cos(
        abs(vecs[1, 0]) / np.linalg.norm(vecs[:, 0])) * 180 / np.pi
    area = np.pi * scale[0] * scale[1]

    return np.mean(ml_acc), np.mean(ap_acc), scale[0]*2, scale[1]*2, angle, principal_dir, area

def calculate_metrics(ml_acc: np.ndarray, ap_acc: np.ndarray) -> Tuple:
    rms_ml = np.sqrt(np.mean(ml_acc**2))
    rms_ap = np.sqrt(np.mean(ap_acc**2))
    total_deviation = np.sum(np.sqrt(ml_acc**2 + ap_acc**2))
    avg_x, avg_y, width, height, angle, direction, _ = calculate_ellipse(
        ml_acc, ap_acc)
    ellipse_area = np.pi * width * height / 4
    return rms_ml, rms_ap, total_deviation, ellipse_area, avg_x, avg_y, width, height, angle, direction

def spectrum_plot(ml, ap, fs):
    n = len(ml)

    # Calcula a FFT
    fft_ml = np.fft.fft(ml)
    fft_ap = np.fft.fft(ap)

    # Calcula as frequências correspondentes
    freqs = np.fft.fftfreq(n, d=1/fs)

    # Pega apenas a parte positiva do espectro
    positive_freqs = freqs[:n//2]
    # Densidade espectral de potência ML
    psd_ml = np.abs(fft_ml[:n//2])**2 / (n*fs)

    # Densidade espectral de potência AP
    psd_ap = np.abs(fft_ap[:n//2])**2 / (n*fs)

    psd_ml = psd_ml[1:]
    psd_ap = psd_ap[1:]
    positive_freqs = positive_freqs[1:]

    return positive_freqs, psd_ml, psd_ap
 
def processar_ytest(df, df2, startRec, endRec, sel, output, filter):
    # Aqui você pode aplicar filtros, normalizações, cálculo de deslocamentos etc.
    df_proc = df.copy()
    df2_proc = df2.copy()
    
    time_vec = df_proc["Tempo"]
    x = df_proc["X"]
    y = df_proc["Y"]
    z = df_proc["Z"]

    ap = z
    if np.mean(x) > np.mean(y):
        ml = y
        v = x

    else:
        ml = x
        v = y

    # Converter tempo para segundos
    t_original = time_vec / 1000  # ms para s

    # Criar vetor de tempo para 100 Hz
    fs_novo = 100  # Hz
    dt_novo = 1 / fs_novo
    t_novo = np.arange(t_original.iloc[0], t_original.iloc[-1], dt_novo)

    # Interpoladores
    interp_ap = interp1d(t_original, ap, kind='linear',
                         fill_value="extrapolate")
    interp_ml = interp1d(t_original, ml, kind='linear',
                         fill_value="extrapolate")
    interp_v = interp1d(t_original, v, kind='linear',
                         fill_value="extrapolate")

    # Sinais reamostrados
    ap_interp_100Hz = interp_ap(t_novo)
    ml_interp_100Hz = interp_ml(t_novo)
    v_interp_100Hz = interp_v(t_novo)

    # 1. Remover tendência
    ap_detrended = detrend(ap_interp_100Hz)
    ml_detrended = detrend(ml_interp_100Hz)
    v_detrended = detrend(v_interp_100Hz)

    # 2. Filtro Butterworth passa-baixa com parâmetros da sidebar
    fs = 100  # Hz
    cutoff = filter  # Usando valor da sidebar
    order = 4  # Usando valor da sidebar

    # Normaliza a frequência de corte (Nyquist = fs/2)
    nyquist = fs / 2
    normal_cutoff = cutoff / nyquist

    # Coeficientes do filtro
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # Aplicando o filtro com zero-phase (filtfilt)
    ap_filtrado = filtfilt(b, a, ap_detrended)
    ml_filtrado = filtfilt(b, a, ml_detrended)
    v_filtrado = filtfilt(b, a, v_detrended)
    
    #joelho
    time_vec_2 = df2_proc["Tempo"]
    x_2 = df2_proc["X"]
    y_2 = df2_proc["Y"]
    z_2 = df2_proc["Z"]

    ap2 = x_2
    v2 = y_2
    ml2 = z_2

    # Converter tempo para segundos
    t_original_2 = time_vec_2 / 1000  # ms para s

    # Criar vetor de tempo para 100 Hz
    fs_novo = 100  # Hz
    dt_novo = 1 / fs_novo
    t_novo2 = np.arange(t_original_2.iloc[0], t_original_2.iloc[-1], dt_novo)

    # Interpoladores
    interp_ap2 = interp1d(t_original_2, ap2, kind='linear',
                         fill_value="extrapolate")
    interp_ml2 = interp1d(t_original_2, ml2, kind='linear',
                         fill_value="extrapolate")
    interp_v2 = interp1d(t_original_2, v2, kind='linear',
                         fill_value="extrapolate")

    # Sinais reamostrados
    ap2_interp_100Hz = interp_ap(t_novo2)
    ml2_interp_100Hz = interp_ml(t_novo2)
    v2_interp_100Hz = interp_v(t_novo2)

    # 1. Remover tendência
    ap2_detrended = detrend(ap2_interp_100Hz)
    ml2_detrended = detrend(ml2_interp_100Hz)
    v2_detrended = detrend(v2_interp_100Hz)

    # 2. Filtro Butterworth passa-baixa com parâmetros da sidebar
    fs = 100  # Hz
    cutoff = filter  # Usando valor da sidebar
    order = 4  # Usando valor da sidebar

    # Normaliza a frequência de corte (Nyquist = fs/2)
    nyquist = fs / 2
    normal_cutoff = cutoff / nyquist

    # Coeficientes do filtro
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # Aplicando o filtro com zero-phase (filtfilt)
    ap2_filtrado = filtfilt(b, a, ap2_detrended)
    ml2_filtrado = filtfilt(b, a, ml2_detrended)
    v2_filtrado = filtfilt(b, a, v2_detrended)
        
    if sel == 1:
        positive_freqs, psd_ml, psd_ap = spectrum_plot(
            ml_filtrado[startRec:endRec], ap_filtrado[startRec:endRec], fs)
        rms_ml, rms_ap, total_deviation, ellipse_area, avg_x, avg_y, width, height, angle, direction = calculate_metrics(
            ml_filtrado[startRec:endRec], ap_filtrado[startRec:endRec])
    else:
        positive_freqs, psd_ml, psd_ap = spectrum_plot(
            ml_filtrado, ap_filtrado, fs)

    if output == 0:
        return t_novo, ml_filtrado, ap_filtrado, v_filtrado, t_novo2, ml2_filtrado, ap2_filtrado, v2_filtrado, positive_freqs, psd_ml, psd_ap
    else:
        return rms_ml, rms_ap, total_deviation, ellipse_area, avg_x, avg_y, width, height, angle, direction
