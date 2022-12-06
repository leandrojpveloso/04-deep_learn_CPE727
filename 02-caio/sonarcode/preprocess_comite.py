from __future__ import division
import warnings
from scipy.signal import decimate, hanning, convolve, spectrogram, lfilter, cheby2, butter, cheb2ord, hilbert
from librosa import stft, fft_frequencies, frames_to_time
import numpy as np
import math
import scipy.signal
from fractions import Fraction

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

def tpsw(signal, npts=None, n=None, p=None, a=None):
    x = np.copy(signal)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if npts is None:
        npts = x.shape[0]
    if n is None:
        n=int(round(npts*.04/2.0+1))
    if p is None:
        p =int(round(n / 8.0 + 1))
    if a is None:
        a = 2.0
    if p>0:
        h = np.concatenate((np.ones((n-p+1)), np.zeros(2 * p-1), np.ones((n-p+1))), axis=None)
    else:
        h = np.ones((1, 2*n+1))
        p = 1
    h /= np.linalg.norm(h, 1)

    def apply_on_spectre(xs):
        return convolve(h, xs, mode='full')

    mx = np.apply_along_axis(apply_on_spectre, arr=x, axis=0)
    ix = int(np.floor((h.shape[0] + 1)/2.0)) # Defasagem do filtro
    mx = mx[ix-1:npts+ix-1] # Corrige da defasagem
    # Corrige os pontos extremos do espectro
    ixp = ix - p
    mult=2*ixp/np.concatenate([np.ones(p-1)*ixp, range(ixp,2*ixp + 1)], axis=0)[:, np.newaxis] # Correcao dos pontos extremos
    mx[:ix,:] = mx[:ix,:]*(np.matmul(mult, np.ones((1, x.shape[1])))) # Pontos iniciais
    mx[npts-ix:npts,:]=mx[npts-ix:npts,:]*np.matmul(np.flipud(mult),np.ones((1, x.shape[1]))) # Pontos finais
    #return mx
    # Elimina picos para a segunda etapa da filtragem
    #indl= np.where((x-a*mx) > 0) # Pontos maiores que a*mx
    indl = (x-a*mx) > 0
    #x[indl] = mx[indl]
    x = np.where(indl, mx, x)
    mx = np.apply_along_axis(apply_on_spectre, arr=x, axis=0)
    mx=mx[ix-1:npts+ix-1,:]
    #Corrige pontos extremos do espectro
    mx[:ix,:]=mx[:ix,:]*(np.matmul(mult,np.ones((1, x.shape[1])))) # Pontos iniciais
    mx[npts-ix:npts,:]=mx[npts-ix:npts,:]*(np.matmul(np.flipud(mult),np.ones((1,x.shape[1])))) # Pontos finais

    if signal.ndim == 1:
        mx = mx[:, 0]
    return mx
     
def lofar(data, dec, fs, n_pts_fft=1024, n_overlap=0,
          spectrum_bins_left=None, **tpsw_args):

    if not isinstance(data, np.ndarray):
        raise NotImplementedError
    
    if dec == 1: # decimação do sinal
        data = data.copy()
    else:
        if data.ndim == 2: # temporary fix for stereo audio. 
            data = data.mean(axis=1)
            data = data.squeeze()
        data = decimate(data, int(dec), ftype='fir', zero_phase=False)
        
    freq, time, power = spectrogram(data,
                                    window=('hann'),
                                    nperseg=n_pts_fft,
                                    noverlap=n_overlap,
                                    nfft=n_pts_fft,
                                    fs=fs,
                                    detrend=False,
                                    axis=0,
                                    scaling='spectrum',
                                    mode='magnitude')
    #For stereo, without further changes, the genreated spectrogram has shape (freq, channel, time)
    if power.ndim == 3: # temporary fix for stereo audio. 
        power = power.mean(axis=1)
        power = power.squeeze()

    power = np.absolute(power)
    power = power / tpsw(power)#, **tpsw_args)
    power = np.log10(power)
    power[power < -0.2] = 0

    if spectrum_bins_left is None:
        spectrum_bins_left = power.shape[0]*0.8
    power = power[:spectrum_bins_left, :]
    freq = freq[:spectrum_bins_left]

    return np.transpose(power), freq, time

def demon(data, fs, n_fft=1024, max_freq=35, overlap_ratio=0.5, apply_bandpass=True, bandpass_specs=None, method='abs'):
    if not isinstance(data, np.ndarray):
        raise ValueError("Input must be of type numpy.ndarray. %s was passed" % type(data))
        
    # transformar os dados de stereo para mono
    if data.ndim == 2: # temporary fix for stereo audio.
        data = data.mean(axis=1)
        data = data.squeeze()
    
    x = data.copy()
    
    # print('dado de entrada: ', x.shape)
        
    # first_pass_sr = 1250 # 31250/25
    first_pass_sr = round(fs/25)
    
    q1 = round(fs/first_pass_sr) # 25 for 31250 sample rate ; decimatio ratio for 1st pass
    q2 = round((fs/q1)/(2*max_freq)) # decimatio ratio for 2nd pass

    fft_over = math.floor(n_fft-2*max_freq*overlap_ratio)

    if apply_bandpass:
        if bandpass_specs is None:
            nyq = fs/2
            wp = [1000/nyq, 2000/nyq]
            ws = [700/nyq, 2300/nyq]
            rp = 0.5
            As = 50
        elif isinstance(bandpass_specs, dict):
            try:
                fp = bandpass_specs["fp"]
                fs = bandpass_specs["fs"]

                # wp = np.array(fp)/nyq
                # ws = np.array(fs)/nyq
                
                wp = np.array(fp)/(fs/2)
                ws = np.array(fs)/(fs/2)
                
                rp = bandpass_specs["rs"]
                As = bandpass_specs["As"]
            except KeyError as e:
                raise KeyError("Missing %s specification for bandpass filter" % e)
        else:
            raise ValueError("bandpass_specs must be of type dict. %s was passed" % type(bandpass_specs))
        
        N, wc = cheb2ord(wp, ws, rp, As) 
        b, a = cheby2(N, rs=As, Wn=wc, btype='bandpass', output='ba', analog=True)
        x = lfilter(b, a, x, axis=0)

    if method=='hilbert':
        x = hilbert(x)
    elif method=='abs':
        x = np.abs(x) # demodulation
    else:
        raise ValueError("Method not found")

    x = decimate(x, q1, ftype='fir', zero_phase=False)
    x = decimate(x, q2, ftype='fir', zero_phase=False)

    final_fs = (fs//q1)//q2

    x /= x.max()
    x -= np.mean(x)
    sxx = stft(x,
               window=('hann'),
               win_length=n_fft,
               hop_length=(n_fft - fft_over),
               n_fft=n_fft)
    freq = fft_frequencies(sr=final_fs, n_fft=n_fft)
    time = frames_to_time(np.arange(0, sxx.shape[1]), 
                   sr=final_fs, hop_length=(n_fft - fft_over))

    sxx = np.absolute(sxx)
    
    sxx = sxx / tpsw(sxx)

    sxx, freq = sxx[8:, :], freq[8:] # ??

    return np.transpose(sxx), freq, time
    
def spectro(data, fs, n_pts_fft=1024, n_overlap=0,
          spectrum_bins_left=None, **tpsw_args):

    if not isinstance(data, np.ndarray):
        raise NotImplementedError

    freq, time, power = spectrogram(data,
                                    window=('hann'),
                                    nperseg=n_pts_fft,
                                    noverlap=n_overlap,
                                    nfft=n_pts_fft,
                                    fs=fs,
                                    detrend=False,
                                    axis=0,
                                    scaling='spectrum',
                                    mode='magnitude')
    #For stereo, without further changes, the genreated spectrogram has shape (freq, channel, time)
    if power.ndim == 3: # temporary fix for stereo audio. 
        power = power.mean(axis=1)
        power = power.squeeze()

    power = np.absolute(power)
    power = 20*np.log10(power)

    return np.transpose(power), freq, time
    
def freq_bins_cutoff(Sxx, fs, target_fs):
  raise NotImplementedError


def preprocess_rawdata24(data, param):
      return (
            data
              # .apply(lambda rr: rr['signal'])
              .apply(lambda rr: resample(rr['signal'], rr['fs'], 
                                         final_fs = param.fs))
              .apply(lofar,
                     param.decimate,
                     param.fs//param.decimate,
                     param.n_fft,
                     param.overlap,
                     param.bins
                    )
      )

def preprocess_rawdata31(data, param):
      return (
            data.apply(lambda rr: rr['signal'])
              .apply(lofar,
                     param.decimate,
                     param.fs//param.decimate,
                     param.n_fft,
                     param.overlap,
                     param.bins
                    )
      )

def preprocess_rawdatademon24(data, param):
      return (
            data
              .apply(lambda rr: resample(rr['signal'], rr['fs'], 
                                         final_fs = param.fs))
              .apply(demon,
                     param.fs,
                     param.n_fft,
                     param.freq_max,
                     param.overlap,
                     param.apply_band
                    )
      )

def preprocess_rawdatademon31(data, param):
      return (
            data.apply(lambda rr: rr['signal'])
              .apply(demon,
                     param.fs,
                     param.n_fft,
                     param.freq_max,
                     param.overlap,
                     param.apply_band
                    )
      )
