import os
import csv
import sys
import shutil


os.makedirs('dataset/Split2')
z=5   #how many instruments
#max accuracy is 10%
train_procent=60/100
val_procent=20/100
test_procent=20/100

with open('analyzed_data2.csv', mode='r') as csv_file:
    reader=csv.reader(csv_file)
    instrument_list=[]
    instrument_number=[]
    for row in reader:
        instrument_list.append(row[0])
        instrument_number.append(row[1])
    instrument_list.pop(0)
    instrument_number.pop(0)
    samples=(int(instrument_number[z])-(int(instrument_number[z]) % 10))
    division=[samples*train_procent,samples*(train_procent+val_procent),samples]
    k=0
    for instrument in instrument_list:
        os.makedirs(f"dataset/Split2/{instrument}")
        os.makedirs(f"dataset/Split2/{instrument}/train")
        os.makedirs(f"dataset/Split2/{instrument}/val")
        os.makedirs(f"dataset/Split2/{instrument}/test")
        c=0
        for root_dir, cur_dir, files in os.walk(r'dataset/all-samples/'):
            for file in files:
                if str(file).endswith('chunk_resampled.wav'):
                    if str(file).startswith(instrument):
                        if c < division[0]:
                            shutil.copy2(f"{root_dir}{file}", f"dataset/Split2/{instrument}/train/{file}")
                            c=c+1
                        if c<division[1] and c>=division[0] :
                            shutil.copy2(f"{root_dir}{file}", f"dataset/Split2/{instrument}/val/{file}")
                            c=c+1
                        if c<division[2] and c>=division[1] :
                            shutil.copy2(f"{root_dir}{file}", f"dataset/Split2/{instrument}/test/{file}")
                            c=c+1
        k=k+1
        if k > z:
            sys.exit()