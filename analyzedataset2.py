import os
import re
import csv 


total=0
labels={}
for root_dir, cur_dir, files in os.walk(r'dataset/all-samples/'):
    for file in files:
        if str(file).endswith('chunk_resampled.wav'):
                total += 1
                name=re.split('_', str(file))[0]
                if name in labels:
                    labels[name] += 1
                else:
                    labels[name] = 1
print('file count:', total)

sorted_dict = dict(sorted(labels.items(), key=lambda x: x[1], reverse=True))
with open('analyzed_data2.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Class', 'Number of files'])
    for key, value in sorted_dict.items():
        writer.writerow([key, value])

# Check file content
with open('analyzed_data2.csv', mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)