#https://www.section.io/engineering-education/machine-learning-for-audio-classification/
#https://colab.research.google.com/drive/1iLMmBnLazIhWBOpnsVfo7lVaQnB3WONv#scrollTo=C16LwhgY8JZ6
import csv
import os
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.math import confusion_matrix
from tensorflow import concat
import matplotlib.pyplot as plt
import seaborn as sns


data_dict = dict()
with open('mcff_results_.csv', mode='r') as csv_file:
    reader=csv.reader(csv_file)
    next(reader)
    for row in reader:
        features_list = list()
        for elem in row[0][1:-1].split(','):
            features_list.append(float(elem))
        data_dict[row[1]] = features_list

train_set = list()
val_set = list()
test_set = list()
for root_dir, cur_dir, files in os.walk(r'dataset/Split1'):
    for file in files:
        if root_dir.endswith('val'):
            val_set.append([data_dict[file], file.split('_')[0]])
        if root_dir.endswith('train'):
            train_set.append([data_dict[file], file.split('_')[0]])
        if root_dir.endswith('test'):
            test_set.append([data_dict[file], file.split('_')[0]])

val_set_df=pd.DataFrame(val_set, columns=['feature','class'])
val_X=np.array(val_set_df['feature'].tolist())
val_y=np.array(val_set_df['class'].tolist())
labelencoder=LabelEncoder()
val_y=to_categorical(labelencoder.fit_transform(val_y))

train_set_df=pd.DataFrame(train_set, columns=['feature','class'])
train_X=np.array(train_set_df['feature'].tolist())
train_y=np.array(train_set_df['class'].tolist())
labelencoder=LabelEncoder()
train_y=to_categorical(labelencoder.fit_transform(train_y))

test_set_df=pd.DataFrame(test_set, columns=['feature','class'])
test_X=np.array(test_set_df['feature'].tolist())
test_y=np.array(test_set_df['class'].tolist())
labelencoder=LabelEncoder()
test_y=to_categorical(labelencoder.fit_transform(test_y))

#print(val_X.shape)
#print(train_X.shape)
#print(val_y.shape)
#print(train_y.shape)

num_labels=train_y.shape[1]
model=Sequential()
###first layer
model.add(Dense(100,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###second layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###third layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

###final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))

print(model.summary())
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
num_epochs = 10
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.hdf5', verbose=1, save_best_only=True)
start = datetime.now()

history = model.fit(
    train_X,
    train_y,
    batch_size=num_batch_size,
    epochs=num_epochs,
    validation_data=(val_X, val_y),
    callbacks=[checkpointer],
    verbose=1
)

duration = datetime.now() - start
print("Training completed in time: ", duration)
val_accuracy=model.evaluate(val_X, val_y,verbose=0)
print(f'Validation accuracy {val_accuracy[1]}')
test_accuracy=model.evaluate(test_X, test_y,verbose=0)
print(f'Test accuracy {test_accuracy[1]}')

metrics = history.history
fig = plt.figure(figsize=(16,6))
plot1 = plt.subplot(1,2,1)
plot1.set_title('Split1 Result')
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch')
plt.ylabel('Loss [CrossEntropy]')

plot2 = plt.subplot(1,2,2)
plot2.set_title('Split1 Result')
plt.plot(history.epoch, 100*np.array(metrics['accuracy']), 100*np.array(metrics['val_accuracy']))
plt.legend(['accuracy', 'val_accuracy'])
plt.ylim([0, 100])
plt.xlabel('Epoch')
plt.ylabel('Accuracy [%]')

predict_x=model.predict(test_X)
y_pred = np.argmax(predict_x, axis=1)
class_data = np.argmax(test_y, axis=1)
confusion_mtx = confusion_matrix(class_data, y_pred)
label_names = labelencoder.classes_
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx,
            xticklabels=label_names,
            yticklabels=label_names,
            annot=True, fmt='g')
plt.xlabel('Prediction Split1')
plt.ylabel('Label')
plt.show()
å