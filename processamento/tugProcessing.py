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


def processar_tug(df1,df2, filter_cutoff):
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
    z_acc_interp = interp_x_acc(t_novo_acc)
    z_acc_interp = interp_y_acc(t_novo_acc)
    z_acc_interp = interp_z_acc(t_novo_acc)

    # Remover tendência
    x_acc_detrended = detrend(x_acc_interp)
    y_acc_detrended = detrend(y_acc_interp)
    z_acc_detrended = detrend(z_acc_interp)

    # Filtragem passa-baixa
    nyquist = fs_novo / 2
    normal_cutoff = filter_cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    x_acc_filtrado = filtfilt(b, a, x_acc_detrend)
    y_acc_filtrado = filtfilt(b, a, y_acc_detrend)
    z_acc_filtrado = filtfilt(b, a, z_acc_detrend)
    norma_acc_filtrado = np.sqrt(x_acc_filtrado^2+y_acc_filtrado^2+z_acc_filtrado^2)

    df_gyro = df2.copy()
    time_vec_gyro = df_acc["Tempo"]
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
    interp_z_gyro = interp1d(t_original, z_acc, kind='linear', fill_value="extrapolate")
    z_gyro_interp = interp_x_acc(t_novo_gyro)
    z_gyro_interp = interp_y_acc(t_novo_gyro)
    z_gyro_interp = interp_z_acc(t_novo_gyro)

    # Remover tendência
    x_gyro_detrended = detrend(x_gyro_interp)
    y_gyro_detrended = detrend(y_gyro_interp)
    z_gyro_detrended = detrend(z_gyro_interp)

    # Filtragem passa-baixa
    nyquist = fs_novo / 2
    normal_cutoff = filter_cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    x_gyro_filtrado = filtfilt(b, a, x_gyro_detrend)
    y_gyro_filtrado = filtfilt(b, a, y_gyro_detrend)
    z_gyro_filtrado = filtfilt(b, a, z_gyro_detrend)
    norma_gyro_filtrado = np.sqrt(x_gyro_filtrado^2+y_gyro_filtrado^2+z_gyro_filtrado^2)
    
    return t_novo_acc, x_acc_filtrado, y_acc_filtrado, z_acc_filtrado, norma_acc_filtrado, t_novo_gyro, x_gyro_filtrado, y_gyro_filtrado, z_gyro_filtrado, norma_gyro_filtrado
