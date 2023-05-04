import os
import librosa
import librosa.display
import csv
import numpy as np

with open('mcff_results_.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['samples', 'File name'])
    for root_dir, cur_dir, files in os.walk(r'dataset/all-samples/'):
        for file in files:
            if str(file).endswith('chunk_resampled.wav'):
                dt, sr = librosa.load(f"{root_dir}/{file}", sr=44100)
                y = librosa.feature.mfcc(y=dt, sr=sr, n_mfcc=40)
                mfccs_scaled_features = np.mean(y.T,axis=0)
                line_list = list()
                line_list.append(list(mfccs_scaled_features))
                line_list.append(file)
                writer.writerow(line_list)
            