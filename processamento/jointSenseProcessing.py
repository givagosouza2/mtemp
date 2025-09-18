import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import math
from scipy.optimize import minimize_scalar

def processar_jps(df, filter):
    # === Função de filtro passa-baixa ===
    def butter_lowpass_filter(data, cutoff_freq, sample_rate, order=4):
        nyquist_freq = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist_freq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, data)
        return filtered_data


    # Aqui você pode aplicar filtros, normalizações, cálculo de deslocamentos etc.
    df_proc = df.copy()
   
    time_vec = df_proc["Tempo"]
    x = df_proc["X"]
    y = df_proc["Y"]
    z = df_proc["Z"]


    # Corrigir tempo para segundos se estiver em milissegundos
    if np.max(tempo) > 1000:
        tempo = tempo / 1000.0

    # Verifica frequência amostral
    amostras_por_segundo = 1 / np.mean(np.diff(tempo))
    

    # Filtros
    cutoff_frequency = 40  # Hz
    sample_rate = 100.0    # Hz
    x_vf = butter_lowpass_filter(x, cutoff_frequency, sample_rate)
    y_vf = butter_lowpass_filter(y, cutoff_frequency, sample_rate)
    z_vf = butter_lowpass_filter(z, cutoff_frequency, sample_rate)
  
            
    return t_novo, x_vf, y_vf e z_vf
    
