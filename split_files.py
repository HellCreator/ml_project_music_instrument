import os
import csv
import sys
import shutil
import random


# default parameters
labels_count=5
#max accuracy is 10%
val_procent=20/100
test_procent=20/100
# Parse console parameters
try:
    if len(sys.argv) > 3:
        labels_count = int(sys.argv[1])
        val_procent = int(sys.argv[2])/100
        test_procent = int(sys.argv[3])/100
        if val_procent + test_procent > 99:
            raise Exception("Sum of parameters > 99 no trainong data")
except Exception as e:
    print(f'failed to parse arguments please pass <label count> <train percent> <validation percent> <test percent> Error {e}')
    sys.exit()

if not os.path.exists('dataset/Split1'):
    os.makedirs('dataset/Split1')
with open('analyzed_data2.csv', mode='r') as csv_file:
    reader=csv.reader(csv_file)
    instruments = {}
    next(reader)
    for row in reader:
        instruments[row[0]] = row[1]

try:
    selected_labels = {k: instruments[k] for k in list(instruments)[:labels_count]}
except Exception as e:
    print(f'failed to select labels Error {e}')
    sys.exit()

min_sample_count = 0
instruments_dict = {}
for instrument in selected_labels.keys():
    if not os.path.exists(f'dataset/Split1/{instrument}'):
        os.makedirs(f"dataset/Split1/{instrument}")
        os.makedirs(f"dataset/Split1/{instrument}/train")
        os.makedirs(f"dataset/Split1/{instrument}/val")
        os.makedirs(f"dataset/Split1/{instrument}/test")
    instrument_files = []
    for root_dir, cur_dir, files in os.walk(r'dataset/all-samples/'):
        for file in files:
            if str(file).endswith('chunk_resampled.wav'):
                if str(file).startswith(instrument):
                    instrument_files.append(f"{root_dir}/{file}")

    random.shuffle(instrument_files)
    samples_count = len(instrument_files)
    for file in instrument_files[:int(samples_count*test_procent)]:
        shutil.copy2(file, f"dataset/Split1/{instrument}/test/{os.path.basename(file)}")
    for file in instrument_files[int(samples_count*test_procent):int(samples_count*test_procent) + int(samples_count*val_procent)]:
        shutil.copy2(file, f"dataset/Split1/{instrument}/val/{os.path.basename(file)}")
    for file in instrument_files[int(samples_count*test_procent) + int(samples_count*val_procent):]:
        shutil.copy2(file, f"dataset/Split1/{instrument}/train/{os.path.basename(file)}")

    print(f'SPLIT1 for label {instrument} '\
        f'test set {len(instrument_files[:int(samples_count*test_procent)])} '\
        f'validation set {len(instrument_files[int(samples_count*test_procent):int(samples_count*test_procent) + int(samples_count*val_procent)])} '\
        f'train set {len(instrument_files[int(samples_count*test_procent) + int(samples_count*val_procent):])} files'
    )
    instruments_dict[instrument] = {'count': samples_count, 'files': instrument_files}
    if min_sample_count == 0 or min_sample_count >= samples_count:
        min_sample_count = samples_count

for instrument in instruments_dict.keys():
    if not os.path.exists(f'dataset/Split2/{instrument}'):
        os.makedirs(f"dataset/Split2/{instrument}")
        os.makedirs(f"dataset/Split2/{instrument}/train")
        os.makedirs(f"dataset/Split2/{instrument}/val")
        os.makedirs(f"dataset/Split2/{instrument}/test")
    instrument_files = instruments_dict[instrument]['files']
    #random.shuffle(instrument_files)
    samples_count = min_sample_count
    for file in instrument_files[:int(samples_count*test_procent)]:
        shutil.copy2(file, f"dataset/Split2/{instrument}/test/{os.path.basename(file)}")
    for file in instrument_files[int(samples_count*test_procent):int(samples_count*test_procent) + int(samples_count*val_procent)]:
        shutil.copy2(file, f"dataset/Split2/{instrument}/val/{os.path.basename(file)}")
    for file in instrument_files[int(samples_count*test_procent) + int(samples_count*val_procent):min_sample_count]:
        shutil.copy2(file, f"dataset/Split2/{instrument}/train/{os.path.basename(file)}")
    print(f'SPLIT2 for label {instrument} '\
        f'test set {len(instrument_files[:int(samples_count*test_procent)])} '\
        f'validation set {len(instrument_files[int(samples_count*test_procent):int(samples_count*test_procent) + int(samples_count*val_procent)])} '\
        f'train set {len(instrument_files[int(samples_count*test_procent) + int(samples_count*val_procent):min_sample_count])} files'
    )

for instrument in instruments_dict.keys():
    instrument_files = instruments_dict[instrument]['files']
    #random.shuffle(instrument_files)
    samples_count = min_sample_count
    if not os.path.exists(f'dataset/Split3/{instrument}'):
        os.makedirs(f"dataset/Split3/{instrument}")
        os.makedirs(f"dataset/Split3/{instrument}/train")
        os.makedirs(f"dataset/Split3/{instrument}/val")
        os.makedirs(f"dataset/Split3/{instrument}/test")
    instrument_files = instruments_dict[instrument]['files']
    #random.shuffle(instrument_files)
    samples_count = min_sample_count
    for file in instrument_files[:int(samples_count*test_procent)]:
        shutil.copy2(file, f"dataset/Split3/{instrument}/test/{os.path.basename(file)}")
    for file in instrument_files[int(samples_count*test_procent):int(samples_count*test_procent) + int(samples_count*val_procent)]:
        shutil.copy2(file, f"dataset/Split3/{instrument}/val/{os.path.basename(file)}")
    for file in instrument_files[int(samples_count*test_procent) + int(samples_count*val_procent):min_sample_count]:
        shutil.copy2(file, f"dataset/Split3/{instrument}/train/{os.path.basename(file)}")

    for fname in os.listdir(f"dataset/Split3/{instrument}/val/"):
        shutil.copy2(os.path.join(f"dataset/Split3/{instrument}/val/",fname), f"dataset/Split3/{instrument}/train/")
    print(f'SPLIT3 for label {instrument} '\
        f'test set {len(instrument_files[:int(samples_count*test_procent)])} '\
        f'validation set {len(instrument_files[int(samples_count*test_procent):int(samples_count*test_procent) + int(samples_count*val_procent)])} '\
        f'train set {len(instrument_files[int(samples_count*test_procent):min_sample_count])} files'
    )

