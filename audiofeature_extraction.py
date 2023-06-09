import sys
import numpy as np
import matplotlib as plt
import librosa
import wave
from pydub import AudioSegment

# Load files
audio_segment = AudioSegment.from_file(sys.argv[1])
# Print attributes
print(f"Channels: {audio_segment.channels}")
print(f"Sample width: {audio_segment.sample_width}")
print(f"Frame rate (sample rate): {audio_segment.frame_rate}")
print(f"Frame width: {audio_segment.frame_width}")
print(f"Length (ms): {len(audio_segment)}")
print(f"Frame count: {audio_segment.frame_count()}")
print(f"Intensity: {audio_segment.dBFS}")


