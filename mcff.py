import os
import librosa
import matplotlib.pyplot as plt
import librosa.display
import matplotlib


for root_dir, cur_dir, files in os.walk(r'dataset/all-samples/'):
    for file in files:
        if str(file).endswith('chunk_resampled.wav'):
            dt, sr = librosa.load(f"{root_dir}{file}", sr=44100)
            y=librosa.feature.mfcc(y=dt, sr=sr, n_fft=1000)
            librosa.display.specshow(y)
            plt.savefig(f"dataset/all-samples/{file}_plot.png")