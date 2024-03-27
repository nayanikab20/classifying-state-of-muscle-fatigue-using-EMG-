import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, square, convolve, welch
from itertools import chain


# function to read in data: Only analyse 1 subject at a time as per objective in PDF
def read_and_structure_data(data_folder, ID):
    data = []

    for root, dirs, files in os.walk(data_folder):
        files = sorted(files)
        # print(files)
        for file in files:
            if file.endswith(".csv"):
                # print(file)

                if (file == "U3Ex2Rep3.csv" or file == "U1Ex1Rep3.csv") and (ID =="U1" or ID=="U3"): # These files do not seem to have data
                    pass
                    print(f'file missing: {file}')

                else:
                    file_path = os.path.join(root, file)

                    # Extract information from the file name
                    user_id = file.split('E')[0]
                    exercise_id = file.split('R')[0][-1]
                    repetition_id = int(file.split('.')[0][-1])

                    if user_id == ID:
                        # print(user_id, file)

                        df = pd.read_csv(file_path)
                        df.columns = ['time', 'raw_emg_data', 'label']

                        # Append data to the list
                        data.append({
                            'user_id': user_id,
                            'exercise_id': exercise_id,
                            'repetition_id': repetition_id,
                            'time': df['time'],
                            'emg_raw_data': df['raw_emg_data'],
                            'label': df['label'] 
                        })

                    # print(f'aa:{data}')

    # Convert to a DataFrame
    structured_data = pd.DataFrame(data)

    # print(data)

    return structured_data

# function to bandpass filter the data
def bandpass_filter(data, sampling_rate, lowcut, highcut, order=4):
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(N=order, Wn=[low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Function to compute RMS in a sliding window
def sliding_window_rms(signal, window_size):
    squared_signal = signal**2
    box = np.ones(window_size)/window_size
    rms = np.sqrt(convolve(squared_signal, box, mode='same'))
    return rms

# function for data augumentation
def extract_windows_and_compute_psd(time_series, window_size=2000, overlap=1000, fs = 2000):
    windows = []
    window_psd = []
    start = 0

    while start + window_size <= len(time_series):
        end = start + window_size
        window = time_series[start:end]
        windows.append(window)
        start += overlap
        frequencies, psd= welch(window, fs=fs, nperseg=256)
        window_psd.append(np.log10(psd))

    return windows, window_psd


# function to extract features
def extract_features(structured_data, fs):

    data = []
    features = []
    labels = []
    for i in range(len(structured_data)):

        filtered_data = bandpass_filter(structured_data['emg_raw_data'][i], fs, 30, 300)
        abs_data = np.abs(filtered_data)
        rms = sliding_window_rms(structured_data['emg_raw_data'][i], 25) # 50 ms window

        nf = filtered_data[structured_data['label'].iloc[i] == 0]
        f = filtered_data[structured_data['label'].iloc[i] == 1]
        
        # Compute FFT
        nf_fft = np.fft.fft(nf)[0:len(nf)//2]
        nf_fft_frequencies = np.fft.fftfreq(len(nf), d=1/2000 )[0:len(nf)//2]

        # Compute FFT
        f_fft = np.fft.fft(f)[0:len(f)//2]
        f_fft_frequencies = np.fft.fftfreq(len(f), d=1/2000 )[0:len(f)//2]

        nf_frequencies, nf_psd= welch(nf, fs=fs, nperseg=256)
        f_frequencies, f_psd= welch(f, fs=fs, nperseg=256)

        # Extract windows with 1000-sample overlap
        nf_trials, nf_trials_psd = extract_windows_and_compute_psd(nf, window_size=2000, overlap=1000)
        f_trials, f_trials_psd = extract_windows_and_compute_psd(f, window_size=2000, overlap=1000)


        data.append({'filtered_data': filtered_data,
                    'abs_data': abs_data,
                    'rms': rms,
                    'nf': nf,
                    'f': f,
                    'nf_fft': [nf_fft, nf_fft_frequencies],
                    'f_fft': [f_fft, f_fft_frequencies],
                    'nf_psd': [np.log10(nf_psd), nf_frequencies],
                    'f_psd': [np.log10(f_psd), f_frequencies]})
        
        # features.append(np.log10(nf_psd)) 
        # features.append(np.log10(f_psd))
        # labels.append(0)
        # labels.append(1)

        features.append(nf_trials_psd)
        features.append(f_trials_psd)
        labels.append([0]*(len(nf_trials_psd)))
        labels.append([1]*(len(f_trials_psd)))
    
    features = list(chain.from_iterable(features))
    labels = list(chain.from_iterable(labels))

    return data, features, labels
