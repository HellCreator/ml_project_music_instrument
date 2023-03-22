# ml_project_music_instrument
ML project classification of sounds
1. https://learn.microsoft.com/en-us/windows/wsl/install (If you use windows)
2. Download dataset https://philharmonia.co.uk/
3. unpack it (we can prepare make file for that)
4. Analyse files according to task 2
5. Normalize Data 
6. augment data (if needed)
7. Convert into MFCC
8. Do splits

Initial setup 
1. clone repo
2. go to directory 
3. install make 
sudo apt-get install make ffmpeg
4. setup venv
python -m venv .
https://docs.python.org/3/library/venv.html
5. activate venv
source bin/activate
6. install dependencies
7. download dataset
make download_dataset
8. unpack dataset
python unpackdataset.py  (use 'python3' instead of 'python' if recently installed python or using latest version)

Additional commands
clean dataset folder
make clean_dataset


Error in ffmpeg unable to decode from mpe to wav (tried two times)
[mp3 @ 0x13ce34370] Failed to read frame size: Could not seek to 1026.
dataset/all-samples/saxophone_Fs3_15_fortissimo_normal.mp3: Invalid argument

[mp3 @ 0x133e34370] Failed to read frame size: Could not seek to 1026.
dataset/all-samples/saxophone_Fs3_15_fortissimo_normal.mp3: Invalid argument


Traceback (most recent call last):
  File "/Users/andriimelnyk/dev/studypg/ml/ml_project_music_instrument/convertsample.py", line 9, in <module>
    mp3_sound = AudioSegment.from_mp3(f"{root_dir}{file}")
  File "/Users/andriimelnyk/dev/studypg/ml/ml_project_music_instrument/lib/python3.9/site-packages/pydub/audio_segment.py", line 796, in from_mp3
    return cls.from_file(file, 'mp3', parameters=parameters)
  File "/Users/andriimelnyk/dev/studypg/ml/ml_project_music_instrument/lib/python3.9/site-packages/pydub/audio_segment.py", line 773, in from_file
    raise CouldntDecodeError(
pydub.exceptions.CouldntDecodeError: Decoding failed. ffmpeg returned error code: 1

Output from ffmpeg/avlib:

ffmpeg version 5.1.2 Copyright (c) 2000-2022 the FFmpeg developers
  built with Apple clang version 14.0.0 (clang-1400.0.29.202)
  configuration: --prefix=/opt/homebrew/Cellar/ffmpeg/5.1.2_6 --enable-shared --enable-pthreads --enable-version3 --cc=clang --host-cflags= --host-ldflags= --enable-ffplay --enable-gnutls --enable-gpl --enable-libaom --enable-libaribb24 --enable-libbluray --enable-libdav1d --enable-libmp3lame --enable-libopus --enable-librav1e --enable-librist --enable-librubberband --enable-libsnappy --enable-libsrt --enable-libsvtav1 --enable-libtesseract --enable-libtheora --enable-libvidstab --enable-libvmaf --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxml2 --enable-libxvid --enable-lzma --enable-libfontconfig --enable-libfreetype --enable-frei0r --enable-libass --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libopenjpeg --enable-libspeex --enable-libsoxr --enable-libzmq --enable-libzimg --disable-libjack --disable-indev=jack --enable-videotoolbox --enable-neon
  libavutil      57. 28.100 / 57. 28.100
  libavcodec     59. 37.100 / 59. 37.100
  libavformat    59. 27.100 / 59. 27.100
  libavdevice    59.  7.100 / 59.  7.100
  libavfilter     8. 44.100 /  8. 44.100
  libswscale      6.  7.100 /  6.  7.100
  libswresample   4.  7.100 /  4.  7.100
  libpostproc    56.  6.100 / 56.  6.100
[mp3 @ 0x133e34370] Failed to read frame size: Could not seek to 1026.
dataset/all-samples/saxophone_Fs3_15_fortissimo_normal.mp3: Invalid argument

deleted file form dataset
rm dataset/all-samples/saxophone_Fs3_15_fortissimo_normal.mp3

[mp3 @ 0x142e34370] Failed to read frame size: Could not seek to 1026.
dataset/all-samples/viola_D6_05_piano_arco-normal.mp3: Invalid argument
