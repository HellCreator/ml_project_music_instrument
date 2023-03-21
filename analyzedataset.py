import os
import re
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
for label in set(labels):
     number=labels.count(label)
     print(f'{label}, {number}')
print(set(labels))
print(len(set(labels)))
#for label in set(labels):
 #    print(f'{label}, number')