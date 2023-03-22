import os
import librosa
from pydub import AudioSegment


for root_dir, cur_dir, files in os.walk(r'dataset/all-samples/'):
    for file in files:
        if str(file).endswith('.mp3'):
            mp3_sound = AudioSegment.from_mp3(f"{root_dir}{file}")
            mp3_sound.export(f"{root_dir}{file}.wav")
            #y, sr = librosa.load(f"{root_dir}{file}.wav", sr=44100)
        
            #librosa.feature.mfcc(y=y, sr=sr)



