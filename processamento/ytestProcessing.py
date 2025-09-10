import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, welch, detrend, savgol_filter
from typing import Tuple

def processar_ytest1(df, filter):
    # Aqui você pode aplicar filtros, normalizações, cálculo de deslocamentos etc.
    df_proc = df.copy()
   
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
            
    return t_novo, ml_filtrado, ap_filtrado, v_filtrado
    
def processar_ytest2(df, filter):
    # Aqui você pode aplicar filtros, normalizações, cálculo de deslocamentos etc.
    df_proc = df.copy()
   
    time_vec = df_proc["Tempo"]
    x = df_proc["X"]
    y = df_proc["Y"]
    z = df_proc["Z"]
    
    ap = x
    v = y
    ml = z

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
            
    return t_novo, ml_filtrado, ap_filtrado, v_filtrado
    
