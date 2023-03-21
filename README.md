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
sudo apt-get install make
4. setup venv
python -m venv .
https://docs.python.org/3/library/venv.html
5. activate venv
source bin/activate
6. download dataset
make download_dataset
7. unpack dataset
python unpackdataset.py  (use 'python3' instead of 'python' if recently installed python or using latest version)

Additional commands
clean dataset folder
make clean_dataset
