import os
from pydub import AudioSegment


for root_dir, cur_dir, files in os.walk(r'dataset/all-samples/'):
    for file in files:
        if str(file).endswith('.mp3'):
            if f'{file}.wav' in files:
                continue # skip already converted files

            mp3_sound = AudioSegment.from_mp3(f"{root_dir}/{file}")
            mp3_sound.export(f"{root_dir}/{file}.wav")

