import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import butter, filtfilt, detrend, savgol_filter
from scipy.integrate import cumulative_trapezoid
from scipy.signal import find_peaks

def top2_peaks(y, x=None, distance=None, prominence=None, abs_peaks=False):
    """
    Retorna índices (e opcionalmente tempos) dos dois maiores picos locais.
    - distance: mínima separação entre picos (em pontos)
    - prominence: 'altura' mínima relativa (ajuda a filtrar ruído)
    - abs_peaks: se True, encontra picos em |y| (para considerar negativos)
    """
    y_use = np.abs(y) if abs_peaks else y
    idx, props = find_peaks(y_use, distance=distance, prominence=prominence)
    if idx.size == 0:
        return [], [] if x is not None else []
    # ordenar por altura do pico (valor em y_use)
    order = np.argsort(y_use[idx])[::-1]  # decrescente
    idx_top = idx[order[:2]].tolist()
    # ordenar pelo tempo se quiser (opcional)
    if x is not None:
        t_top = [x[i] for i in idx_top]
        return idx_top, t_top
    return idx_top

def processar_tug(df1,df2,filter_cutoff1,filter_cutoff2):
    df_acc = df1.copy()
    time_vec_acc = df_acc["Tempo"]
    x_acc = df_acc["X"]
    y_acc = df_acc["Y"]
    z_acc = df_acc["Z"]

    # Reamostragem para 100 Hz
    t_original = time_vec_acc / 1000  # Converter para segundos
    fs_novo = 100
    dt_novo = 1 / fs_novo
    t_novo_acc = np.arange(t_original.iloc[0], t_original.iloc[-1], dt_novo)

    # Interpolação
    interp_x_acc = interp1d(t_original, x_acc, kind='linear', fill_value="extrapolate")
    interp_y_acc = interp1d(t_original, y_acc, kind='linear', fill_value="extrapolate")
    interp_z_acc = interp1d(t_original, z_acc, kind='linear', fill_value="extrapolate")
    x_acc_interp = interp_x_acc(t_novo_acc)
    y_acc_interp = interp_y_acc(t_novo_acc)
    z_acc_interp = interp_z_acc(t_novo_acc)

    if np.mean(x_acc_interp) > np.mean(y_acc_interp):
        smartphone = 0
    else:
        smartphone = 1

    # Remover tendência
    x_acc_detrended = detrend(x_acc_interp)
    y_acc_detrended = detrend(y_acc_interp)
    z_acc_detrended = detrend(z_acc_interp)

    # Filtragem passa-baixa
    nyquist = fs_novo / 2
    normal_cutoff = filter_cutoff1 / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    x_acc_filtrado = filtfilt(b, a, x_acc_detrended)
    y_acc_filtrado = filtfilt(b, a, y_acc_detrended)
    z_acc_filtrado = filtfilt(b, a, z_acc_detrended)
    norma_acc_filtrado = np.sqrt(x_acc_filtrado**2+y_acc_filtrado**2+z_acc_filtrado**2)

    df_gyro = df2.copy()
    time_vec_gyro = df_gyro["Tempo"]
    x_gyro = df_gyro["X"]
    y_gyro = df_gyro["Y"]
    z_gyro = df_gyro["Z"]

    # Reamostragem para 100 Hz
    t_original = time_vec_gyro / 1000  # Converter para segundos
    fs_novo = 100
    dt_novo = 1 / fs_novo
    t_novo_gyro = np.arange(t_original.iloc[0], t_original.iloc[-1], dt_novo)

    # Interpolação
    interp_x_gyro = interp1d(t_original, x_gyro, kind='linear', fill_value="extrapolate")
    interp_y_gyro = interp1d(t_original, y_gyro, kind='linear', fill_value="extrapolate")
    interp_z_gyro = interp1d(t_original, z_gyro, kind='linear', fill_value="extrapolate")
    x_gyro_interp = interp_x_gyro(t_novo_gyro)
    y_gyro_interp = interp_y_gyro(t_novo_gyro)
    z_gyro_interp = interp_z_gyro(t_novo_gyro)

    # Remover tendência
    x_gyro_detrended = detrend(x_gyro_interp)
    y_gyro_detrended = detrend(y_gyro_interp)
    z_gyro_detrended = detrend(z_gyro_interp)

    # Filtragem passa-baixa
    nyquist = fs_novo / 2
    normal_cutoff = filter_cutoff2 / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    x_gyro_filtrado = filtfilt(b, a, x_gyro_detrended)
    y_gyro_filtrado = filtfilt(b, a, y_gyro_detrended)
    z_gyro_filtrado = filtfilt(b, a, z_gyro_detrended)
    norma_gyro_filtrado = np.sqrt(x_gyro_filtrado**2+y_gyro_filtrado**2+z_gyro_filtrado**2)

    if smartphone == 0:
        v_gyro = x_gyro_filtrado
        ml_gyro = y_gyro_filtrado
        v_acc = x_acc_filtrado
        ml_acc = y_acc_filtrado
    else:
        ml_gyro = x_gyro_filtrado
        v_gyro = y_gyro_filtrado
        v_acc = y_acc_filtrado
        ml_acc = x_acc_filtrado

    for index, valor in enumerate(np.sqrt(ml_gyro**2)):
        if valor > 0.25:
            start_test = t_novo_gyro[index]
            break
            
    for index in range(len(ml_gyro) - 1, -1, -1):
        valor = np.sqrt(ml_gyro[index]**2)
        if valor > 0.25:
            stop_test = t_novo_gyro[index]
            break
    idx_top = top2_peaks(np.sqrt(v_gyro**2), x=t_novo_gyro, distance=None, prominence=None, abs_peaks=False)
    idx_top_ml = top2_peaks(np.sqrt(ml_gyro**2), x=t_novo_gyro, distance=None, prominence=None, abs_peaks=False)
    idx_top_acc_ap = top2_peaks(np.sqrt(z_acc_filtrado**2), x=t_novo_acc, distance=None, prominence=None, abs_peaks=False)
    idx_top_acc_v = top2_peaks(np.sqrt(y_acc_filtrado**2), x=t_novo_acc, distance=None, prominence=None, abs_peaks=False)
        
    duration = stop_test - start_test
    
    return t_novo_acc, v_acc, ml_acc, z_acc_filtrado, norma_acc_filtrado, t_novo_gyro, v_gyro, ml_gyro, z_gyro_filtrado, norma_gyro_filtrado, start_test, stop_test,idx_top,idx_top_ml,idx_top_acc_ap,idx_top_acc_v,duration
