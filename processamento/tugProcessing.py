import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import butter, filtfilt, detrend, savgol_filter
from scipy.integrate import cumulative_trapezoid

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
    else:
        ml_gyro = x_gyro_filtrado
        v_gyro = y_gyro_filtrado

    for index, valor in enumerate(ml_gyro):
        if valor > 0.5:
            start_test = t_novo_gyro[index]
            break
            
    for index in range(len(ml_gyro) - 1, -1, -1):
        valor = ml_gyro[index]
        if valor > 0.5:
            stop_test = t_novo_gyro[index]
            break
    return t_novo_acc, x_acc_filtrado, y_acc_filtrado, z_acc_filtrado, norma_acc_filtrado, t_novo_gyro, v_gyro, ml_gyro, z_gyro_filtrado, norma_gyro_filtrado, start_test, stop_test
