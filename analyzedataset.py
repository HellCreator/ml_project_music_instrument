import os
import re
import csv 


audiofiles=0
count = 0
labels=[]
for root_dir, cur_dir, files in os.walk(r'dataset/all-samples/'):
    count += len(files)
    for file in files:
        if str(file).endswith('.mp3'):
                audiofiles=audiofiles+1
                name=re.split('_', str(file))
                labels.append(name[0])
print('file count:', audiofiles)

my_dict = dict(map(lambda x: (x, labels.count(x)), labels))
sorted_dict = dict(sorted(my_dict.items(), key=lambda x: x[1], reverse=True))
with open('analyzed_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Class', 'Number of files'])
    for key, value in sorted_dict.items():
        writer.writerow([key, value])

with open('analyzed_data.csv', mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)