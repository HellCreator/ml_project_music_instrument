'''
this script:
convert files from mp3 to wav, 
split it into 10s samples 
convert into mfcc data using librosa
'''
'''
from pydub import AudioSegment

# files                                                                         
src = "dataset/all-samples/banana shaker/bass drum"
dst = "dataset/convertet_samples/test.wav"

# convert wav to mp3                                                            
sound = AudioSegment.from_mp3(src)
sound.export(dst, format="wav")
'''
#########################

import os
path = "dataset/all-samples"
dir_list = os.listdir(path)