import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa

# for baseline 
import parselmouth 
from parselmouth.praat import call
import fathon
from fathon import fathonUtils as fu
# for RPDE
from pyrpde import  rpde
# for wavelet 
import pywt
from tqwt_tools import tqwt
import dtcwt
from scipy.fftpack import dct
# for EMD
from PyEMD import EMD
# for embeddings
import transformers
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, Wav2Vec2FeatureExtractor, AutoConfig
# for GQ
from scipy.signal import lfilter, find_peaks
# hilbert
from scipy.signal import hilbert

import statistics
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import entropy

import os
import re
import json
import pandas as pd

data_dir = './PD/AudioKoreaDrKim'

patient_dirs = sorted(os.listdir(data_dir))[1:-3]
# print(patient_dirs)
# print(len(patient_dirs))

processor = AutoProcessor.from_pretrained("steja/whisper-small-korean")
model = AutoModelForSpeechSeq2Seq.from_pretrained("steja/whisper-small-korean")

patient_dict = {}
for patient in patient_dirs:
    patient_dir = os.path.join(data_dir, patient)
    patient_recs = os.listdir(patient_dir)
    [_, id, ord, age, gender, date, no] = patient_recs[0].split('-')
    patient_dict[patient] = {'id':id, 'ord':ord, 'age':age, 'gender':gender, 'date':date, 'recordings':{}}
    for rec in patient_recs:
        patient_dict[patient]['recordings'][(re.search('_(.*).wav', rec).group(1))]={}
        
        patient_dict[patient]['recordings'][(re.search('_(.*).wav', rec).group(1))]['duration'] = librosa.get_duration(path=os.path.join(patient_dir, rec))

        audio_data, sample_rate = librosa.load(os.path.join(patient_dir, rec), sr=16000)

        input_features = processor(audio_data, sampling_rate=sample_rate, return_tensors="pt").input_features
        predicted_ids = model.generate(input_features)

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        patient_dict[patient]['recordings'][(re.search('_(.*).wav', rec).group(1))]['whisper-small-korean'] = transcription
    # print(patient_dict[patient]['recordings'])

data_dir = './PD/AudioKoreaDrKim'
def data_to_path(data, pat):
    paths = []
    prefix = '{}/{}/{}-{}-{}-{}-{}-{}_'.format(data_dir, pat, pat.split(' ')[0], data['ord'], data['age'], data['gender'], data['date'], 'NC' if 'NC' in pat else 'PD')
    for rec in data['recordings']:
        paths.append(prefix+rec.split(':')[0]+'.wav')

    return paths

def measureIntensity(sound,  pitch_floor=75.):
    intensity = call(sound, 'To Intensity', pitch_floor, 0., 'yes')
    minIntensity = call(intensity, 'Get minimum',0., 0., "Parabolic")
    maxIntensity = call(intensity, 'Get maximum', 0., 0., "Parabolic")
    meanIntensity = call(intensity, 'Get mean', 0., 0.)
    stddevIntensity = call(intensity, 'Get standard deviation', 0., 0.)
    medianIntensity = call(intensity, 'Get quantile', 0., 0., 0.50)

    return minIntensity, maxIntensity, meanIntensity, stddevIntensity, medianIntensity

def measurePitch(sound, f0min, f0max, unit):
    duration = call(sound, "Get total duration") # duration
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
    meanF0 = call(pitch, "Get mean", 0., 0., unit) # get mean pitch
    medianF0 = call(pitch, 'Get quantile', 0., 0., 0.50, unit)
    stdevF0 = call(pitch, "Get standard deviation", 0 ,0, unit) # get standard deviation
    minF0 = call(pitch, 'Get minimum', 0., 0., unit, "Parabolic")
    maxF0 = call(pitch, 'Get maximum', 0., 0., unit, "Parabolic")
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
    meanHnr = call(harmonicity, "Get mean", 0, 0)
    minHnr = call(harmonicity, 'Get minimum', 0., 0., "Parabolic")
    maxHnr = call(harmonicity, 'Get maximum', 0., 0., "Parabolic")
    stdevHnr = call(harmonicity, 'Get standard deviation', 0., 0.)\

    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    return duration, meanF0, medianF0, stdevF0, minF0, maxF0, meanHnr, minHnr, maxHnr, stdevHnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer

def measureFormants(sound, f0min,f0max):
    pitch = call(sound, "To Pitch (cc)", 0, f0min, 15, 'no', 0.03, 0.45, 0.01, 0.35, 0.14, f0max)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    
    formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
    numPoints = call(pointProcess, "Get number of points")

    f1_list = []
    f2_list = []
    f3_list = []
    f4_list = []
    
    # Measure formants only at glottal pulses
    for point in range(0, numPoints):
        point += 1
        t = call(pointProcess, "Get time from index", point)
        f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        f4 = call(formants, "Get value at time", 4, t, 'Hertz', 'Linear')
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)
        f4_list.append(f4)
    
    f1_list = [f1 for f1 in f1_list if str(f1) != 'nan']
    f2_list = [f2 for f2 in f2_list if str(f2) != 'nan']
    f3_list = [f3 for f3 in f3_list if str(f3) != 'nan']
    f4_list = [f4 for f4 in f4_list if str(f4) != 'nan']
    
    # calculate mean formants across pulses
    f1_mean = statistics.mean(f1_list)
    f2_mean = statistics.mean(f2_list)
    f3_mean = statistics.mean(f3_list)
    f4_mean = statistics.mean(f4_list)
    
    # calculate median formants across pulses, this is what is used in all subsequent calcualtions
    # you can use mean if you want, just edit the code in the boxes below to replace median with mean
    f1_median = statistics.median(f1_list)
    f2_median = statistics.median(f2_list)
    f3_median = statistics.median(f3_list)
    f4_median = statistics.median(f4_list)
    
    return f1_mean, f2_mean, f3_mean, f4_mean, f1_median, f2_median, f3_median, f4_median

def measureMFCC(path, number_of_mfcc_features = 12):
    x, sr = librosa.load(path, sr=16000)
    mean_mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=number_of_mfcc_features).T,axis=0).tolist()
    std_mfccs = np.std(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=number_of_mfcc_features).T,axis=0).tolist()
    var_mfccs = np.var(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=number_of_mfcc_features).T,axis=0).tolist()
    min_mfccs = np.amin(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=number_of_mfcc_features).T,axis=0).tolist()
    max_mfccs = np.amax(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=number_of_mfcc_features).T,axis=0).tolist()
    return mean_mfccs, std_mfccs, var_mfccs, min_mfccs, max_mfccs

def measureGNE(sound,  minimum_frequency=500., maximum_frequency=4500.,
                                          bandwidth=1000., step=80.):
    matrix = call(sound, "To Harmonicity (gne)", minimum_frequency, maximum_frequency, bandwidth, step)
    min_gne = call(matrix, 'Get minimum')
    max_gne = call(matrix, 'Get maximum')
    mean_gne = call(matrix, 'Get mean...', 0., 0., 0., 0.)
    stddev_gne = call(matrix, 'Get standard deviation...', 0., 0., 0., 0.)
    sum_gne = call(matrix, 'Get sum')


    return mean_gne, sum_gne, stddev_gne, min_gne, max_gne

def get_spectrum_attributes(sound, band_floor=200., band_ceiling=1000., low_band_floor=0.,
                            low_band_ceiling=500., high_band_floor=500., high_band_ceiling=4000.,
                            power=2., moment=3., return_values=False, replacement_for_nan=0.):
    spectrum = call(sound, 'To Spectrum', 'yes')
    band_energy = call(spectrum, 'Get band energy', band_floor, band_ceiling)
    band_density = call(spectrum, 'Get band density', band_floor, band_ceiling)
    band_energy_difference = call(spectrum, 'Get band energy difference', low_band_floor, low_band_ceiling, high_band_floor, high_band_ceiling)
    band_density_difference = call(spectrum, 'Get band density difference', low_band_floor, low_band_ceiling, high_band_floor, high_band_ceiling)
    center_of_gravity_spectrum = call(spectrum, 'Get centre of gravity', power)
    stddev_spectrum = call(spectrum, 'Get standard deviation', power)
    skewness_spectrum = call(spectrum, 'Get skewness', power)
    kurtosis_spectrum = call(spectrum, 'Get kurtosis',power)
    central_moment_spectrum = call(spectrum, 'Get central moment', moment, power)

    return band_energy, band_density, band_energy_difference, band_density_difference, center_of_gravity_spectrum, stddev_spectrum, skewness_spectrum, kurtosis_spectrum, central_moment_spectrum

def get_formant_attributes(sound, time_step=0., pitch_floor=75., pitch_ceiling=600.,
                           max_num_formants=5., max_formant=5500.,
                           window_length=0.025, pre_emphasis_from=50.,
                           unit='Hertz', interpolation_method='Linear', replacement_for_nan=0.):
    
    point_process = call(sound, "To PointProcess (periodic, cc)", pitch_floor, pitch_ceiling)
    formant = call(sound, "To Formant (burg)", time_step, max_num_formants, max_formant, window_length, pre_emphasis_from)
    num_points = call(point_process, "Get number of points")
    f1_list, f2_list, f3_list, f4_list = [], [], [], []

    # Measure formants only at glottal pulses
    for point in range(1, num_points+1):
        t = call(point_process, "Get time from index", point)
        f1 = call(formant, "Get value at time", 1, t, unit, interpolation_method)
        f2 = call(formant, "Get value at time", 2, t, unit, interpolation_method)
        f3 = call(formant, "Get value at time", 3, t, unit, interpolation_method)
        f4 = call(formant, "Get value at time", 4, t, unit, interpolation_method)
        f1_list.append(f1 if not np.isnan(f1) else replacement_for_nan)
        f2_list.append(f2 if not np.isnan(f2) else replacement_for_nan)
        f3_list.append(f3 if not np.isnan(f3) else replacement_for_nan)
        f4_list.append(f4 if not np.isnan(f4) else replacement_for_nan)


    # Calculate mean formants across pulses
    f1_mean = statistics.mean(f1_list)
    f2_mean = statistics.mean(f2_list)
    f3_mean = statistics.mean(f3_list)
    f4_mean = statistics.mean(f4_list)

    # Calculate median formants across pulses
    f1_median = statistics.median(f1_list)
    f2_median = statistics.median(f2_list)
    f3_median = statistics.median(f3_list)
    f4_median = statistics.median(f4_list)

    # Formant Dispersion (Fitch, W. T. (1997). Vocal tract length and formant frequency
    # dispersion correlate with body size in rhesus macaques. The Journal of the Acoustical
    # Society of America, 102(2), 1213-1222.)
    formant_dispersion = (f4_median - f1_median) / 3

    # Average Formant (Pisanski, K., & Rendall, D. (2011). The prioritization of voice
    # fundamental frequency or formants in listeners’ assessments of speaker size, masculinity,
    # and attractiveness. The Journal of the Acoustical Society of America, 129(4), 2201-2212.)
    average_formant = (f1_median + f2_median + f3_median + f4_median) / 4

    # MFF (Smith, D. R., & Patterson, R. D. (2005). The interaction of glottal-pulse rate and
    # vocal-tract length in judgements of speaker size, sex, and age. The Journal of the
    # Acoustical Society of America, 118(5), 3177-3186.)
    mff = (f1_median * f2_median * f3_median * f4_median) ** 0.25

    # Fitch VTL (Fitch, W. T. (1997). Vocal tract length and formant frequency dispersion
    # correlate with body size in rhesus macaques. The Journal of the Acoustical Society of
    # America, 102(2), 1213-1222.)
    fitch_vtl = ((1 * (35000 / (4 * f1_median))) +
                 (3 * (35000 / (4 * f2_median))) +
                 (5 * (35000 / (4 * f3_median))) +
                 (7 * (35000 / (4 * f4_median)))) / 4

    # Delta F (Reby, D., & McComb, K.(2003). Anatomical constraints generate honesty: acoustic
    # cues to age and weight in the roars of red deer stags. Animal Behaviour, 65, 519e-530.)
    xy_sum = ((0.5 * f1_median) +
              (1.5 * f2_median) +
              (2.5 * f3_median) +
              (3.5 * f4_median))
    x_squared_sum = (0.5 ** 2) + (1.5 ** 2) + (2.5 ** 2) + (3.5 ** 2)
    delta_f = xy_sum / x_squared_sum

    # VTL(Delta F) Reby, D., & McComb, K.(2003).Anatomical constraints generate honesty: acoustic
    # cues to age and weight in the roars of red deer stags. Animal Behaviour, 65, 519e-530.)
    vtl_delta_f = 35000 / (2 * delta_f)

    return formant_dispersion, average_formant, mff, fitch_vtl, delta_f, vtl_delta_f

def extract_dwt_features(data, wavelet, level=10):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    features = []
    for coeff in coeffs:
        features.append(np.mean(coeff))
        features.append(np.median(coeff))
        features.append(np.std(coeff))
        features.append(np.min(coeff))
        features.append(np.max(coeff))
    return features


def extract_tqwt_features(data, Q, r, J):
    coeffs = tqwt(data, Q, r, J)
    features = []
    for coeff in coeffs:
        real_coeff = coeff.real
        imag_coeff = coeff.imag
        features.append(np.mean(real_coeff))
        features.append(np.median(real_coeff))
        features.append(np.std(real_coeff))
        features.append(np.min(real_coeff))
        features.append(np.max(real_coeff))
        features.append(np.mean(imag_coeff))
        features.append(np.median(imag_coeff))
        features.append(np.std(imag_coeff))
        features.append(np.min(imag_coeff))
        features.append(np.max(imag_coeff))
    return features

def extract_dtcwt_features(data, level=10):
    transform = dtcwt.Transform1d()
    coeffs = transform.forward(data, nlevels=level)
    features = []
    for i in range(level):
        real_coeff = coeffs.highpasses[i].real
        imag_coeff = coeffs.highpasses[i].imag
        features.append(np.mean(real_coeff))
        features.append(np.median(real_coeff))
        features.append(np.std(real_coeff))
        features.append(np.min(real_coeff))
        features.append(np.max(real_coeff))
        features.append(np.mean(imag_coeff))
        features.append(np.median(imag_coeff))
        features.append(np.std(imag_coeff))
        features.append(np.min(imag_coeff))
        features.append(np.max(imag_coeff))
    return features

# TQWT parameters
Q = 4  # Tunable Q-factor
r = 3  # Redundancy
J = 10  # Number of levels
wavelet = 'db4'  # Daubechies wavelet

def extract_features(audio_path):
    # Step 1: Load the audio file
    sample_rate, audio_signal = wavfile.read(audio_path)

    # If the audio has two channels (stereo), take one channel
    if len(audio_signal.shape) > 1:
        audio_signal = audio_signal[:, 0]

    # Normalize the signal
    audio_signal = audio_signal / np.max(np.abs(audio_signal))
    # Step 2: Apply Empirical Mode Decomposition (EMD)
    emd = EMD()
    IMFs = emd(audio_signal)

    features = []
    for ind, imf in enumerate(IMFs):
        # Example features: mean, variance, skewness, kurtosis
        mean = np.mean(imf)
        median = np.median(imf)
        std = np.std(imf)
        mn = np.min(imf)
        mx = np.max(imf)
        skewness = pd.Series(imf).skew()
        kurtosis = pd.Series(imf).kurtosis()
        features.extend([mean, median, std, mn, mx, skewness, kurtosis])
    return np.array(features).tolist()

# Function to calculate energy of a signal
def calculate_energy(signal):
    return np.sum(np.square(signal))

# Function to calculate nonlinear energy of a signal (example using absolute value)
def calculate_nonlinear_energy(signal):
    return np.sum(np.abs(signal))

# Function to calculate entropy of a signal
def calculate_entropy(signal):
    prob_distribution = np.histogram(signal, bins=256, density=True)[0]
    return entropy(prob_distribution)

# Function to compute the Vocal Fold Excitation Ratio (VFER)
def compute_vfer(audio_file, frame_length=1024, hop_length=512):
    # Load the audio signal
    signal, sr = librosa.load(audio_file, sr=None)

    # Normalize the signal
    signal = signal / np.max(np.abs(signal))

    # Split signal into frames
    frames = librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length)

    # Initialize lists to hold energy, nonlinear energy, and entropy values
    energies = []
    nonlinear_energies = []
    entropies = []

    for frame in frames.T:
        # Compute energy, nonlinear energy, and entropy for each frame
        energy = calculate_energy(frame)
        nonlinear_energy = calculate_nonlinear_energy(frame)
        entropy_value = calculate_entropy(frame)
        
        energies.append(energy)
        nonlinear_energies.append(nonlinear_energy)
        entropies.append(entropy_value)

    # Convert lists to numpy arrays
    energies = np.array(energies)
    nonlinear_energies = np.array(nonlinear_energies)
    entropies = np.array(entropies)

    # Compute VFER as ratio of energy to the sum of energy, nonlinear energy, and entropy
    vfer_values = energies / (energies + nonlinear_energies + entropies)

    return vfer_values

def extract_glottal_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)

    # Plot the audio signal
    # plt.figure(figsize=(10, 4))
    # plt.plot(y)
    # plt.title("Speech Signal")
    # plt.xlabel("Time (samples)")
    # plt.ylabel("Amplitude")
    # plt.show()

    # Step 2: Estimate the Glottal Flow using LPC
    def lpc_coefficients(y, order=12):
        """Compute LPC coefficients."""
        return librosa.lpc(y, order=order)

    # Get LPC coefficients
    lpc_order = 16  # Typically 8-16 works for speech
    lpc = lpc_coefficients(y, lpc_order)

    # Invert the filter to estimate the glottal waveform
    glottal_flow = lfilter([1], lpc, y)

    # Plot the estimated glottal flow waveform
    # plt.figure(figsize=(10, 4))
    # plt.plot(glottal_flow)
    # plt.title("Estimated Glottal Flow")
    # plt.xlabel("Time (samples)")
    # plt.ylabel("Amplitude")
    # plt.show()

    # Step 3: Detect Glottal Closure and Opening Instants
    # Compute the derivative of the glottal waveform
    glottal_derivative = np.diff(glottal_flow)

    # Detect the glottal closure instants (negative peaks)
    glottal_closure_peaks, _ = find_peaks(-glottal_derivative, height=0.02, distance=sr // 200)  # Example threshold

    # Detect the glottal opening instants (positive peaks)
    glottal_opening_peaks, _ = find_peaks(glottal_derivative, height=0.02, distance=sr // 200)

    # Plot the derivative with detected closure and opening instants
    # plt.figure(figsize=(10, 4))
    # plt.plot(glottal_derivative, label="Glottal Derivative")
    # plt.plot(glottal_closure_peaks, glottal_derivative[glottal_closure_peaks], "x", label="Closure Instants", markersize=10)
    # plt.plot(glottal_opening_peaks, glottal_derivative[glottal_opening_peaks], "o", label="Opening Instants", markersize=10)
    # plt.title("Glottal Derivative with Closure and Opening Instants")
    # plt.xlabel("Time (samples)")
    # plt.ylabel("Amplitude")
    # plt.legend()
    # plt.show()


    # Ensure correct pairing: Each GOI should pair with the next GCI (not the previous one)
    min_length = min(len(glottal_opening_peaks), len(glottal_closure_peaks) - 1)

    # Ensure the GOI is paired with the next GCI
    valid_gois = []
    valid_gcis = []

    for goi in glottal_opening_peaks:
        # Find the next GCI that occurs after this GOI
        next_gc_index = np.where(glottal_closure_peaks > goi)[0]
        if len(next_gc_index) > 0:
            if (len(valid_gcis) > 0 and goi < valid_gcis[-1]):
                valid_gois = valid_gois[:-1]
            else: 
                valid_gcis.append(glottal_closure_peaks[next_gc_index[0]])

            valid_gois.append(goi)

    # Convert valid_gois and valid_gcis to numpy arrays for further calculations
    valid_gois = np.array(valid_gois)
    valid_gcis = np.array(valid_gcis)
    # Calculate Glottal Period (T_cycle) = difference between consecutive GCIs
    glottal_periods = np.diff(valid_gcis) / sr  # Convert samples to time (in seconds)

    # Calculate Open Time (T_open) = difference between GOIs and GCIs
    # Ensure the number of cycles is consistent
    glottal_open_times = (valid_gcis[:-1] - valid_gois[:-1]) / sr

    # Calculate the Open Quotient for each glottal cycle where both GOIs and GCIs are present
    open_quotients = glottal_open_times / glottal_periods
    # print(open_quotients.shape)

    # Plot the glottal features: Glottal Period and Open Quotient
    # plt.figure(figsize=(10, 4))
    # plt.plot(glottal_periods, label="Glottal Period (s)")
    # plt.plot(open_quotients, label="Open Quotient")
    # plt.title("Glottal Features")
    # plt.xlabel("Glottal Cycle")
    # plt.legend()
    # plt.show()

    valid_gois = valid_gois / sr
    if len(valid_gois) == 0:
        valid_gois = np.append(valid_gois, [0], axis=0)
    mean_gois = np.mean(valid_gois)
    median_gois = np.median(valid_gois)
    stdev_gois = np.std(valid_gois)
    min_gois = np.min(valid_gois)
    max_gois = np.max(valid_gois)
    gois_features = [mean_gois, median_gois, stdev_gois, min_gois, max_gois]

    valid_gcis = valid_gcis / sr
    if len(valid_gcis) == 0:
        valid_gcis = np.append(valid_gcis, [0], axis=0)    
    mean_gcis = np.mean(valid_gcis)
    median_gcis = np.median(valid_gcis)
    stdev_gcis = np.std(valid_gcis)
    min_gcis = np.min(valid_gcis)
    max_gcis = np.max(valid_gcis)
    gcis_features = [mean_gcis, median_gcis, stdev_gcis, min_gcis, max_gcis]

    if len(open_quotients) == 0:
        open_quotients = np.append(open_quotients, [0], axis=0)
    mean_op = np.mean(open_quotients)
    median_op = np.median(open_quotients)
    stdev_op = np.std(open_quotients)
    min_op = np.min(open_quotients)
    max_op = np.max(open_quotients)
    op_features = [mean_op, median_op, stdev_op, min_op, max_op]

    return gois_features, gcis_features, op_features

def mel_filter_bank(sr, n_fft, n_mels=26):
    """ Create a Mel filter bank. """
    return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)

def imf_to_cepstral(imf, sr, n_mels=26, n_fft=2048):
    """ Convert a single IMF to cepstral coefficients """
    # Step 1: Compute the FFT of the IMF
    spectrum = np.abs(np.fft.fft(imf, n=n_fft))[:n_fft//2+1]
    
    # Step 2: Apply Mel-filter bank
    mel_fb = mel_filter_bank(sr, n_fft, n_mels)
    mel_spectrum = np.dot(mel_fb, spectrum)
    
    # Step 3: Take the logarithm of the Mel spectrum
    log_mel_spectrum = np.log(mel_spectrum + 1e-8)  # Avoid log(0)
    
    # Step 4: Apply Discrete Cosine Transform (DCT)
    cepstral_coeffs = dct(log_mel_spectrum, type=2, norm='ortho')
    
    return cepstral_coeffs[:13]  # Typically keep 13 coefficients

def frame_signal(signal, frame_size, hop_size, sr):
    """Frame the signal with a specific frame size and hop size"""
    return librosa.util.frame(signal, frame_length=frame_size, hop_length=hop_size).T

def compute_instantaneous_energy(frames):
    """Compute the energy for each frame"""
    return np.sum(frames ** 2, axis=1)

def compute_energy_deviation(energy):
    """Compute the deviation of energy from frame to frame"""
    return np.diff(energy)

def mel_filter_bank(sr, n_fft, n_mels=26):
    """ Create a Mel filter bank. """
    return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)

def energy_deviation_to_cepstral(energy_deviation, sr, n_mels=26, n_fft=2048):
    """ Convert energy deviation to cepstral coefficients """
    # Step 1: Compute the FFT of the energy deviation
    spectrum = np.abs(np.fft.fft(energy_deviation, n=n_fft))[:n_fft//2+1]
    
    # Step 2: Apply Mel-filter bank
    mel_fb = mel_filter_bank(sr, n_fft, n_mels)
    mel_spectrum = np.dot(mel_fb, spectrum)
    
    # Step 3: Take the logarithm of the Mel spectrum
    log_mel_spectrum = np.log(mel_spectrum + 1e-8)  # Avoid log(0)
    
    # Step 4: Apply Discrete Cosine Transform (DCT)
    cepstral_coeffs = dct(log_mel_spectrum, type=2, norm='ortho')
    
    return cepstral_coeffs[:13]  # Typically keep 13 coefficients

def compute_iedcc(signal, sr, frame_size=1024, hop_size=512, n_mels=26, n_fft=2048):
    """ Compute IEDCC for a given signal """
    # Step 1: Frame the signal
    frames = frame_signal(signal, frame_size, hop_size, sr)
    
    # Step 2: Compute instantaneous energy
    energy = compute_instantaneous_energy(frames)
    
    # Step 3: Compute energy deviation
    energy_deviation = compute_energy_deviation(energy)
    
    # Step 4: Compute cepstral coefficients from energy deviation
    iedcc = energy_deviation_to_cepstral(energy_deviation, sr, n_mels, n_fft)
    
    return iedcc

# Step 1: Empirical Mode Decomposition (EMD)
def perform_emd(signal):
    """ Decompose the signal using EMD into IMFs """
    emd = EMD()
    imfs = emd(signal)
    return imfs

# Step 2: Hilbert Transform to get instantaneous frequency and amplitude
def hilbert_transform(imf):
    """ Apply the Hilbert Transform to an IMF """
    analytic_signal = hilbert(imf)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi)
    
    return amplitude_envelope, instantaneous_frequency

# Step 3: Hilbert-Huang Transform (HHT)
def hht(signal):
    """ Perform HHT by applying Hilbert Transform to the IMFs obtained from EMD """
    imfs = perform_emd(signal)
    amplitudes = []
    frequencies = []
    
    for imf in imfs:
        amp, freq = hilbert_transform(imf)
        amplitudes.append(amp)
        frequencies.append(freq)
    
    return imfs, amplitudes, frequencies


feature_names = ["Duration", "MeanF0", "MedianF0", "StdevF0", "MinF0", "MaxF0", 
                 "MeanIntensity", "MedianIntensity", "StdevIntensity", "MinIntensity", "MaxIntensity"
                 "MeanHnr", "minHnr", "MaxHnr", "StdevHnr",
                 "LocalJitter", "LocalabsoluteJitter", "RapJitter", "Ppq5Jitter", "DdpJitter",
                 "LocalShimmer", "LocaldbShimmer", "Apq3Shimmer", "Aqpq5Shimmer", "Apq11Shimmer", "DdaShimmer",
                 "MeanMFCC", "StdMFCC", "VarMFCC", "MinMFCC", "MaxMFCC",
                 "MeanGNE", "SumGNE", "StdevGNE", "MinGNE", "MaxGNE", 
                 "bandEnergy", "bandDensity", "bandEnergyDifference", "bandDensityDifference", "centerOfGravitySpectrum", "stddevSpectrum", 
                 "skewnessSpectrum", "kurtosisSpectrum", "centralMomentSpectrum",
                 "formantDispersion", "averageFormant", "MFF", "fitchVTL", "deltaF", "VTLDeltaF",
                 "meanDWT", "medianDWT", "StdevDWT", "MinDWT", "MaxDWT", 
                 "meanTQWT", "medianTQWT", "StdevTQWT", "MinTQWT", "MaxTQWT", 
                 "meanDTCWT", "medianDTCWT", "StdevDTCWT", "MinDTCWT", "MaxDTCWT",
                 "meanEMD", "medianEMD", "StdevEMD", "MinEMD", "MaxEMD", "SkewnessEMD", "KurtosisEMD",
                 "meanVFER", "medianVFER", "StdevVFER", "MinVFER", "MaxVFER",
                 "meanGOIS", "medianGOIS", "StdevGOIS", "MinGOIS", "MaxGOIS",
                 "meanGCIS", "medianGCIS", "StdevGCIS", "MinGCIS", "MaxGCIS",
                 "meanOP", "medianOP", "StdevOP", "MinOP", "MaxOP",
                 "imfcc(17,13)", "iedcc(13)", 
                 "hilbert_amp(119)", "hilbert_freq(119)",
                 ]

input_embeddings = {}
for pat, val in patient_dict.items():
    input_embeddings[pat] = {}
    paths = data_to_path(val, pat)
    print(pat)
    for path in paths:

        snd = parselmouth.Sound(path)
        # baseline 1
        (duration, meanF0, medianF0, stdevF0, minF0, maxF0, meanHnr, minHnr, maxHnr, stdevHnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, 
            localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer) = measurePitch(snd, 75, 600, "Hertz")
        
        (minIntensity, maxIntensity, meanIntensity, stddevIntensity, medianIntensity) = measureIntensity(snd)

        (f1_mean, f2_mean, f3_mean, f4_mean, f1_median, f2_median, f3_median, f4_median) = measureFormants(snd, 75, 600)

        # baseline 2
        y, sr = librosa.load(path, sr=None)
        
        (mean_gne, sum_gne, stddev_gne, min_gne, max_gne) = measureGNE(snd)

        (band_energy, band_density, band_energy_difference, band_density_difference, center_of_gravity_spectrum, 
         stddev_spectrum, skewness_spectrum, kurtosis_spectrum, central_moment_spectrum) = get_spectrum_attributes(snd)
        
        (formant_dispersion, average_formant, mff, fitch_vtl, delta_f, vtl_delta_f) = get_formant_attributes(snd)

        entropy, histogram = rpde(y)

        y_normalized = (y - np.mean(y)) / np.std(y)
        time_series = fu.toAggregated(y_normalized)

        total_length = len(y)
        min_window_size = 4
        max_window_size = total_length // 10 
        scales = fu.linRangeByStep(min_window_size, max_window_size, 2)

        # Perform DFA
        dfa = fathon.DFA(time_series)
        lag, dfa_values = dfa.computeFlucVec(scales)
        alpha = dfa.fitFlucVec()

        # MFCC
        mean_mfccs, std_mfccs, var_mfccs, min_mfccs, max_mfccs = measureMFCC(path)

        # wavelet
        sampling_rate, data = wavfile.read(path)
        data = data / np.max(np.abs(data))
        
        dwt_features = extract_dwt_features(data, wavelet)
        # print(len(dwt_features))

        # Extract simulated TQWT features
        tqwt_features = extract_tqwt_features(data, Q, r, J)
        # print(len(tqwt_features))

        # Extract DTCWT features
        dtcwt_features = extract_dtcwt_features(data)

        # Combine features
        qt_features = dwt_features + tqwt_features + dtcwt_features
        # print(len(features))

        # # Convert to DataFrame for better visualization
        dwt_columns = [f'{stat}DWT_{i}' for i in range(len(dwt_features) // 5) for stat in ['mean', 'median', 'std', 'min', 'max']]
        tqwt_columns = [f'{stat}TQWT_{i}' for i in range(len(tqwt_features) // 10) for stat in ['mean_real', 'median_real', 'std_real', 'min_real', 'max_real', 'mean_imag', 'median_imag',  'std_imag', 'min_imag', 'max_imag']]
        dtcwt_columns = [f'{stat}DTCWT_{i}' for i in range(len(dtcwt_features) // 10) for stat in ['mean_real', 'median_real', 'std_real', 'min_real', 'max_real', 'mean_imag', 'median_imag',  'std_imag', 'min_imag', 'max_imag']]

        qt_columns = dwt_columns + tqwt_columns + dtcwt_columns
        qt_features_df = pd.DataFrame([qt_features], columns=qt_columns)

        # EMD
        emd_features = extract_features(path)
        emd_columns = [f'{stat}EMD_{i}' for i in range(len(emd_features) // 7) for stat in ['mean', 'median', 'std', 'min', 'max', 'skewness', 'kurtosis']]
        
        emd_features_df = pd.DataFrame([emd_features], columns=emd_columns)

        # VFER
        vfer_values = compute_vfer(path)
        mean_vfer = np.mean(vfer_values)
        median_vfer = np.median(vfer_values)
        stdev_vfer = np.std(vfer_values)
        min_vfer = np.min(vfer_values)
        max_vfer = np.max(vfer_values)

        # GQ
        gois_features, gcis_features, op_features = extract_glottal_features(path)

        # IMFCC
        sr, audio_signal = wavfile.read(path)

        # If the audio has two channels (stereo), take one channel
        if len(audio_signal.shape) > 1:
            audio_signal = audio_signal[:, 0]

        # Normalize the signal
        signal = audio_signal / np.max(np.abs(audio_signal))
        # Perform EMD to get IMFs
        emd = EMD()
        imfs = emd(signal)

        imfccs = []
        for imf in imfs[:17]:
            imfccs.append(imf_to_cepstral(imf, sr))

        # Convert to numpy array for easier manipulation
        imfccs = np.array(imfccs)

        # print(imfccs.shape)
        imfcc_features = imfccs.flatten().tolist()
        # print(len(imfcc_features))
        imfcc_columns = [f'imfcc_{imf_id}_{mfcc_id}' for imf_id in range(1, 18) for mfcc_id in range(1, 14)]
        imfcc_features_df = pd.DataFrame([imfcc_features], columns=imfcc_columns)

        # IEDCC
        # Load an example audio signal
        signal, sr = librosa.load('../pd_model/PD/AudioKoreaDrKim/1-001/1-001-LBU-70-M-20220704-PD_014.wav', sr=None)

        # Compute IEDCC
        iedcc = compute_iedcc(signal, sr)
        
        iedcc_columns = [f'iedcc_{mfcc_id}'for mfcc_id in range(1, 14)]
        iedcc_features_df = pd.DataFrame([iedcc], columns=imfcc_columns)

        # hilbert
        # load signal
        sample_rate, audio_signal = wavfile.read('../pd_model/PD/AudioKoreaDrKim/1-001/1-001-LBU-70-M-20220704-PD_014.wav')

        # If the audio has two channels (stereo), take one channel
        if len(audio_signal.shape) > 1:
            audio_signal = audio_signal[:, 0]

        # Normalize the signal
        signal = audio_signal / np.max(np.abs(audio_signal))
        
        # Perform HHT
        imfs, amplitudes, frequencies = hht(signal)
        
        amp_features = []
        for i, amp in enumerate(amplitudes):
            mean = np.mean(amp)
            median = np.median(amp)
            std = np.std(amp)
            mn = np.min(amp)
            mx = np.max(amp)
            skewness = pd.Series(amp).skew()
            kurtosis = pd.Series(amp).kurtosis()
            amp_features.extend([mean, median, std, mn, mx, skewness, kurtosis])

        freq_features = []
        for i, freq in enumerate(frequencies):
            mean = np.mean(freq)
            median = np.median(freq)
            std = np.std(freq)
            mn = np.min(freq)
            mx = np.max(freq)
            skewness = pd.Series(freq).skew()
            kurtosis = pd.Series(freq).kurtosis()
            freq_features.extend([mean, median, std, mn, mx, skewness, kurtosis])

        amp_columns = [f'{stat}HIL_AMP_{i}' for i in range(len(amp_features) // 7) for stat in ['mean', 'median', 'std', 'min', 'max', 'skewness', 'kurtosis']]
        freq_columns = [f'{stat}HIL_FREQ_{i}' for i in range(len(freq_features) // 7) for stat in ['mean', 'median', 'std', 'min', 'max', 'skewness', 'kurtosis']]

        amp_features_df = pd.DataFrame([amp_features], columns=amp_columns)
        freq_features_df = pd.DataFrame([freq_features], columns=freq_columns)

        input_embeddings[pat][path[-7:-4]] = {"Duration":duration, "MeanF0":meanF0, "MedianF0":medianF0, "StdevF0":stdevF0, "MinF0":minF0, "MaxF0":maxF0, 
                                              "MeanIntensity":meanIntensity, "MedianIntensity":medianIntensity, "StdevIntensity":stddevIntensity, "MinIntensity":minIntensity, "MaxIntensity":maxIntensity,
                                              "MeanHNR":meanHnr, "MinHnr": minHnr, "MaxHnr":maxHnr, "StdevHnr":stdevHnr, 
                                              "LocalJitter":localJitter, "LocalabsoluteJitter":localabsoluteJitter, "RapJitter":rapJitter, "Ppq5Jitter":ppq5Jitter, "DdpJitter":ddpJitter, 
                                              "LocalShimmer":localShimmer, "LocaldbShimmer":localdbShimmer, "Apq3Shimmer":apq3Shimmer, 
                                              "Aqpq5Shimmer":aqpq5Shimmer, "Apq11Shimmer":apq11Shimmer, "DdaShimmer":ddaShimmer,
                                              "MeanMFCC":mean_mfccs, "StdMFCC":std_mfccs, "VarMFCC":var_mfccs, "MinMFCC":min_mfccs, "MaxMFCC":max_mfccs,
                                              "MeanGNE":mean_gne, "SumGNE":sum_gne, "StdevGNE":stddev_gne, "MinGNE":min_gne, "MaxGNE":max_gne,
                                              "bandEnergy":band_energy, "bandDensity":band_density, "bandEnergyDifference":band_energy_difference, "bandDensityDifference":band_density_difference, 
                                              "centerOfGravitySpectrum":center_of_gravity_spectrum, "stddevSpectrum":stddev_spectrum, "skewnessSpectrum": skewness_spectrum, 
                                              "kurtosisSpectrum": kurtosis_spectrum, "centralMomentSpectrum":central_moment_spectrum,
                                              "formantDispersion": formant_dispersion, "averageFormant": average_formant, "MFF": mff, 
                                              "fitchVTL":fitch_vtl, "deltaF": delta_f, "VTLDeltaF": vtl_delta_f,
                                              "entropyRPDE":entropy, "histogramRPDE":histogram, 
                                              "alphaDFA":alpha, "meanDFA":np.mean(dfa_values), "stdevDFA":np.std(dfa_values), "minDFA":np.min(dfa_values), "maxDFA":np.max(dfa_values),
                                              "meanVFER":mean_vfer, "medianVFER":median_vfer, "StdevVFER":stdev_vfer, "MinVFER":min_vfer, "MaxVFER":max_vfer,
                                              "meanGOIS":gois_features[0], "medianGOIS":gois_features[1], "StdevGOIS":gois_features[2], "MinGOIS":gois_features[3], "MaxGOIS":gois_features[4],
                                              "meanGCIS":gcis_features[0], "medianGCIS":gcis_features[1], "StdevGCIS":gcis_features[2], "MinGCIS":gcis_features[3], "MaxGCIS":gcis_features[4],
                                              "meanOP":op_features[0], "medianOP":op_features[1], "StdevOP":op_features[2], "MinOP":op_features[3], "MaxOP":op_features[4],

                                              }
        
        for key, val in qt_features_df.to_dict().items():
            input_embeddings[pat][path[-7:-4]][key] = val[0]

        for key, val in emd_features_df.to_dict().items():
            input_embeddings[pat][path[-7:-4]][key] = val[0]

        for key, val in imfcc_features_df.to_dict().items():
            input_embeddings[pat][path[-7:-4]][key] = val[0]
        
        for key, val in iedcc_features_df.to_dict().items():
            input_embeddings[pat][path[-7:-4]][key] = val[0]

        for key, val in amp_features_df.to_dict().items():
            input_embeddings[pat][path[-7:-4]][key] = val[0]

        for key, val in freq_features_df.to_dict().items():
            input_embeddings[pat][path[-7:-4]][key] = val[0]


# input_embeddings = np.array(input_embeddings)
# print(input_embeddings.shape)

with open("all_features.json", "w") as fp:
    json.dump(input_embeddings, fp)
