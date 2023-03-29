import soundfile as sf
import os
import sys
from pydub.utils import make_chunks


for root_dir, cur_dir, files in os.walk(r'dataset/all-samples/'):
    for file in files:
        if not str(file).endswith('_resampled.wav'):
            if str(file).endswith('.wav'):
                data, samplerate = sf.read(f"{root_dir}/{file}")
                chunks = make_chunks(data, samplerate)
                for i, chunk in enumerate(chunks):
                    if not len(chunk)<44100:
                        sf.write(f"{root_dir}/{file}{i}_chunk_resampled.wav", chunk, 44100)
