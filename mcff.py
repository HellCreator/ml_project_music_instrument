import os
import librosa
import matplotlib.pyplot as plt
import librosa.display
import matplotlib
import csv
import sys


with open('mcff_results_.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['File name', 'samples'])
    for root_dir, cur_dir, files in os.walk(r'dataset/all-samples/'):
        for file in files:
            if str(file).endswith('chunk_resampled.wav'):
                dt, sr = librosa.load(f"{root_dir}/{file}", sr=44100)
                y=librosa.feature.mfcc(y=dt, sr=sr, n_fft=1000)
                line_list= []
                line_list.append(file)
                for element in y:
                    line_list.append(element[0])
                    line_list.append(element[1])
                writer.writerow(line_list)
            