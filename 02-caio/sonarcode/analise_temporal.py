import numpy as np
from scipy.signal import decimate 
from scipy.signal import cheby2, cheb2ord
from scipy import signal
from fractions import Fraction
import scipy

def time_process(data, ftype_input, zero_phase_input, process_config):
  dec = process_config.decimate
  
  # transformar os dados de stereo para mono
  if data.ndim == 2: # temporary fix for stereo audio. 
    data = data.mean(axis=1)
    data = data.squeeze()
  
  if dec == 1: # decimação do sinal
    data_dec = data.copy()
  else:
    data_dec = decimate(data, int(dec), ftype=ftype_input, zero_phase=zero_phase_input)
  
  h = dec_filtro(process_config) # filtro passa baixa
  data_fil = signal.sosfilt(h, data_dec)
  
  if process_config.normalize == True:
    data_norm = (data_fil-data_fil.min())/(data_fil.max()-data_fil.min()) # normalização
  else:
    data_norm = data_fil

  return data_norm

# projetar o filtro
def dec_filtro(process_config):
  dec = process_config.decimate
  Fs = int(process_config.fs/dec) 
  fp = int(process_config.fp/dec)  
  fs = int(process_config.fc/dec)
  Ap = process_config.Ap 
  As = process_config.As
  wp = fp/(Fs/2)  
  ws = fs/(Fs/2)  
  N, wc = signal.cheb2ord(wp, ws, Ap, As)
  filtro = signal.cheby2(N, As, wc, 'low', output='sos')
  return filtro

# Processamento da análise temporal
def preprocess_rawtempo(raw_data, filtro, phase, process_config):
    return (
        raw_data
            .apply(lambda rr: rr['signal'])
            .apply(time_process, filtro, phase, process_config)
    )

def resample(signal, fs, final_fs, window=('kaiser', 5.0)):
        resample_ratio = Fraction(final_fs, fs)

        upsampling_factor = resample_ratio.numerator
        downsampling_factor = resample_ratio.denominator

        resampled_signal = scipy.signal.resample_poly(
            signal, 
            upsampling_factor, 
            downsampling_factor,
            axis=0, 
            window=window
        )

        return resampled_signal
      
  # Processamento da análise temporal
def preprocess_raw24tempo(raw_data, filtro, phase, process_config):
    return (
        raw_data
            .apply(lambda rr: resample(rr['signal'], rr['fs'], 
                                         final_fs = process_config.fs))
            .apply(time_process, filtro, phase, process_config)
    )
