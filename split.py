import os
import csv
import sys
import random


# default parameters
labels_count=5
default_label='other'
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

with open('analyzed_data2.csv', mode='r') as csv_file:
    reader=csv.reader(csv_file)
    instruments = {}
    next(reader)
    for row in reader:
        instruments[row[0]] = row[1]

try:
    selected_labels = {k: instruments[k] for k in list(instruments)[:labels_count]}
    labels = list(selected_labels.keys())
except Exception as e:
    print(f'failed to select labels Error {e}')
    sys.exit()

min_sample_count = 0
instruments_dict = {}
if not os.path.exists('dataset/Split1'):
    os.makedirs('dataset/Split1')
if not os.path.exists('dataset/Split2'):
    os.makedirs('dataset/Split2')
if not os.path.exists('dataset/Split3'):
    os.makedirs('dataset/Split3')

mffc_dict = dict()

with open('mcff_results_.csv', mode='r') as csv_file:
    reader=csv.reader(csv_file)
    next(reader)
    for row in reader:
        mffc_dict[row[1]] = row[0]

labeled_dict = dict()

for label in selected_labels:
    labeled_dict[label] = dict()

labeled_dict[default_label] = dict()

for file in mffc_dict.keys():
    label = file.split('_')[0]
    if label in selected_labels:
        labeled_dict[label][file] = mffc_dict[file]
    else:
        labeled_dict[default_label][file] = mffc_dict[file]

for label in selected_labels:
    myKeys = list(labeled_dict[label].keys())
    myKeys.sort()
    sorted_dict = {i: labeled_dict[label][i] for i in myKeys}
    labeled_dict[label] = sorted_dict

myKeys = list(labeled_dict[default_label].keys())
myKeys.sort()
sorted_dict = {i: labeled_dict[default_label][i] for i in myKeys}
labeled_dict[default_label] = sorted_dict

# SPLIT1
with open('dataset/Split1/test.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file)
    samples_count = len(labeled_dict[default_label])
    default_count = int(samples_count*test_procent)
    for file in list(labeled_dict[default_label].keys())[:default_count]:
        writer.writerow([labeled_dict[default_label][file], default_label])
    count = dict()
    for label in selected_labels:
        samples_count = len(labeled_dict[label])
        count[label] = int(samples_count*test_procent)
        default_count += count[label]
        for file in list(labeled_dict[label].keys())[:count[label]]:
            writer.writerow([labeled_dict[label][file], label])

    print(f"Split1 Testing {default_count}")

with open('dataset/Split1/validate.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file)
    samples_count = len(labeled_dict[default_label])
    default_count = int(samples_count*val_procent)
    for file in list(labeled_dict[default_label].keys())[int(samples_count*test_procent):int(samples_count*test_procent) + int(samples_count*val_procent)]:
        writer.writerow([labeled_dict[default_label][file], default_label])
    count = dict()
    for label in selected_labels:
        samples_count = len(labeled_dict[label])
        default_count += int(samples_count*val_procent)
        for file in list(labeled_dict[label].keys())[int(samples_count*test_procent):int(samples_count*test_procent) + int(samples_count*val_procent)]:
            writer.writerow([labeled_dict[label][file], label])
    print(f"Split1 Validation {default_count}")

with open('dataset/Split1/train.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file)
    samples_count = len(labeled_dict[default_label])
    default_count = samples_count - int(samples_count*test_procent) - int(samples_count*val_procent)
    for file in list(labeled_dict[default_label].keys())[int(samples_count*test_procent) + int(samples_count*val_procent):]:
        writer.writerow([labeled_dict[default_label][file], default_label])
    count = dict()
    for label in selected_labels:
        samples_count = len(labeled_dict[label])
        default_count += samples_count - int(samples_count*test_procent) - int(samples_count*val_procent)
        for file in list(labeled_dict[label].keys())[int(samples_count*test_procent) + int(samples_count*val_procent):]:
            writer.writerow([labeled_dict[label][file], label])
    print(f"Split1 Train {default_count}")

# SPLIT2
min_sample_count = 0
for label in labeled_dict.keys():
    samples_count = len(labeled_dict[label].keys())
    if min_sample_count == 0 or min_sample_count >= samples_count:
        min_sample_count = samples_count

with open('dataset/Split2/test.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file)
    samples_count = min_sample_count
    default_count = int(samples_count*test_procent)
    for file in list(labeled_dict[default_label].keys())[:default_count]:
        writer.writerow([labeled_dict[default_label][file], default_label])
    count = dict()
    for label in selected_labels:
        samples_count = min_sample_count
        count[label] = int(samples_count*test_procent)
        default_count += count[label]
        for file in list(labeled_dict[label].keys())[:count[label]]:
            writer.writerow([labeled_dict[label][file], label])

    print(f"Split2 Testing {default_count}")

with open('dataset/Split2/validate.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file)
    samples_count = min_sample_count
    default_count = int(samples_count*val_procent)
    for file in list(labeled_dict[default_label].keys())[int(samples_count*test_procent):int(samples_count*test_procent) + int(samples_count*val_procent)]:
        writer.writerow([labeled_dict[default_label][file], default_label])
    count = dict()
    for label in selected_labels:
        samples_count = min_sample_count
        default_count += int(samples_count*val_procent)
        for file in list(labeled_dict[label].keys())[int(samples_count*test_procent):int(samples_count*test_procent) + int(samples_count*val_procent)]:
            writer.writerow([labeled_dict[label][file], label])
    print(f"Split2 Validation {default_count}")

with open('dataset/Split2/train.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file)
    samples_count = min_sample_count
    default_count = samples_count - int(samples_count*test_procent) - int(samples_count*val_procent)
    for file in list(labeled_dict[default_label].keys())[int(samples_count*test_procent) + int(samples_count*val_procent):min_sample_count]:
        writer.writerow([labeled_dict[default_label][file], default_label])
    count = dict()
    for label in selected_labels:
        samples_count = min_sample_count
        default_count += samples_count - int(samples_count*test_procent) - int(samples_count*val_procent)
        for file in list(labeled_dict[label].keys())[int(samples_count*test_procent) + int(samples_count*val_procent):min_sample_count]:
            writer.writerow([labeled_dict[label][file], label])
    print(f"Split2 Train {default_count}")

# SPLIT3
with open('dataset/Split3/test.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file)
    samples_count = min_sample_count
    default_count = int(samples_count*test_procent)
    for file in list(labeled_dict[default_label].keys())[:default_count]:
        writer.writerow([labeled_dict[default_label][file], default_label])
    count = dict()
    for label in selected_labels:
        samples_count = min_sample_count
        count[label] = int(samples_count*test_procent)
        default_count += count[label]
        for file in list(labeled_dict[label].keys())[:count[label]]:
            writer.writerow([labeled_dict[label][file], label])

    print(f"Split3 Testing {default_count}")

with open('dataset/Split3/validate.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file)
    samples_count = min_sample_count
    default_count = int(samples_count*val_procent)
    for file in list(labeled_dict[default_label].keys())[int(samples_count*test_procent):int(samples_count*test_procent) + int(samples_count*val_procent)]:
        writer.writerow([labeled_dict[default_label][file], default_label])
    count = dict()
    for label in selected_labels:
        samples_count = min_sample_count
        default_count += int(samples_count*val_procent)
        for file in list(labeled_dict[label].keys())[int(samples_count*test_procent):int(samples_count*test_procent) + int(samples_count*val_procent)]:
            writer.writerow([labeled_dict[label][file], label])
    print(f"Split3 Validation {default_count}")

with open('dataset/Split3/train.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file)
    samples_count = min_sample_count
    default_count = samples_count - int(samples_count*test_procent)
    for file in list(labeled_dict[default_label].keys())[int(samples_count*test_procent):min_sample_count]:
        writer.writerow([labeled_dict[default_label][file], default_label])
    count = dict()
    for label in selected_labels:
        samples_count = min_sample_count
        default_count += samples_count - int(samples_count*test_procent)
        for file in list(labeled_dict[label].keys())[int(samples_count*test_procent):min_sample_count]:
            writer.writerow([labeled_dict[label][file], label])
    print(f"Split3 Train {default_count}")
