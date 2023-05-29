#https://www.section.io/engineering-education/machine-learning-for-audio-classification/
#https://colab.research.google.com/drive/1iLMmBnLazIhWBOpnsVfo7lVaQnB3WONv#scrollTo=C16LwhgY8JZ6
import csv
import os
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.math import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

num_epochs = 30
num_batch_size = 32

def get_data_from_files(path_to_split):
    train_set = list()
    with open(path_to_split + '/train.csv', mode='r') as csv_file:
        reader=csv.reader(csv_file)
        for row in reader:
            features_list = list()
            for elem in row[0][1:-1].split(','):
                features_list.append(float(elem))
            train_set.append([features_list, row[1]])

    val_set = list()
    with open(path_to_split + '/validate.csv', mode='r') as csv_file:
        reader=csv.reader(csv_file)
        for row in reader:
            features_list = list()
            for elem in row[0][1:-1].split(','):
                features_list.append(float(elem))
            val_set.append([features_list, row[1]])

    test_set = list()
    with open(path_to_split + '/test.csv', mode='r') as csv_file:
        reader=csv.reader(csv_file)
        for row in reader:
            features_list = list()
            for elem in row[0][1:-1].split(','):
                features_list.append(float(elem))
            test_set.append([features_list, row[1]])

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
    return val_X, val_y, train_X, train_y, test_X, test_y, labelencoder.classes_

def define_model(labels_count):
    model=Sequential()
    ###first layer
    model.add(Dense(200,input_shape=(128,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    ###second layer
    model.add(Dense(600))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    ###third layer
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    ###final layer
    model.add(Dense(labels_count))
    model.add(Activation('softmax'))
    # optimizer='adam' # good
    # optimizer='Nadam' # better
    # optimizer='Adagrad' # very bad
    # optimizer='Adamax' # good
    # optimizer='RMSprop'# best but confusion matrix is bad
    # optimizer='SGD' # do not work at all
    model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='Nadam')
    return model

def generate_plots(history, name, test_X, test_y):
    metrics = history.history
    fig = plt.figure(figsize=(16,6))
    plot1 = plt.subplot(1,2,1)
    plot1.set_title(f'{name} Result')
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel('Epoch')
    plt.ylabel('Loss [CrossEntropy]')
    plot2 = plt.subplot(1,2,2)
    plot2.set_title(f'{name} Result')
    plt.plot(history.epoch, 100*np.array(metrics['accuracy']), 100*np.array(metrics['val_accuracy']))
    plt.legend(['accuracy', 'val_accuracy'])
    plt.ylim([0, 100])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [%]')
    predict_x=model.predict(test_X, batch_size=num_batch_size)
    y_pred = np.argmax(predict_x, axis=1)
    class_data = np.argmax(test_y, axis=1)
    print(classification_report(y_pred, class_data))
    confusion_mtx = confusion_matrix(class_data, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx,
            xticklabels=label_names,
            yticklabels=label_names,
            annot=True, fmt='g')
    plt.xlabel(f'Prediction {name}')
    plt.ylabel('Label Test dataset')
    
    predict_x=model.predict(val_X, batch_size=num_batch_size)
    y_pred = np.argmax(predict_x, axis=1)
    class_data = np.argmax(val_y, axis=1)
    confusion_mtx = confusion_matrix(class_data, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx,
            xticklabels=label_names,
            yticklabels=label_names,
            annot=True, fmt='g')
    plt.xlabel(f'Prediction {name}')
    plt.ylabel('Label Validation dataset')

    predict_x=model.predict(train_X, batch_size=num_batch_size)
    y_pred = np.argmax(predict_x, axis=1)
    class_data = np.argmax(train_y, axis=1)
    confusion_mtx = confusion_matrix(class_data, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx,
            xticklabels=label_names,
            yticklabels=label_names,
            annot=True, fmt='g')
    plt.xlabel(f'Prediction {name}')
    plt.ylabel('Label Training dataset')


val_X, val_y, train_X, train_y, test_X, test_y, label_names = get_data_from_files(r'dataset/Split1')
model = define_model(len(label_names))
print(model.summary())
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

duration1 = datetime.now() - start
val_accuracy1 = model.evaluate(val_X, val_y,verbose=0)
test_accuracy1=model.evaluate(test_X, test_y,verbose=0)
generate_plots(history, 'Split1', test_X, test_y)
# Split2
val_X, val_y, train_X, train_y, test_X, test_y, label_names = get_data_from_files(r'dataset/Split2')
model1 = define_model(len(label_names))
print(model1.summary())
checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification2.hdf5', verbose=1, save_best_only=True)
start = datetime.now()

history1 = model1.fit(
    train_X,
    train_y,
    batch_size=num_batch_size,
    epochs=num_epochs,
    validation_data=(val_X, val_y),
    callbacks=[checkpointer],
    verbose=1
)

duration2 = datetime.now() - start
val_accuracy2=model1.evaluate(val_X, val_y,verbose=0)
test_accuracy2=model1.evaluate(test_X, test_y,verbose=0)
generate_plots(history1, 'Split2', test_X, test_y)
# Split2 end

val_X, val_y, train_X, train_y, test_X, test_y, label_names = get_data_from_files(r'dataset/Split3')
model2 = define_model(len(label_names))
print(model2.summary())
checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification3.hdf5', verbose=1, save_best_only=True)
start = datetime.now()

history2 = model2.fit(
    train_X,
    train_y,
    batch_size=num_batch_size,
    epochs=num_epochs,
    validation_data=(val_X, val_y),
    callbacks=[checkpointer],
    verbose=1
)
duration3 = datetime.now() - start
val_accuracy3 = model2.evaluate(val_X, val_y,verbose=0)
test_accuracy3=model2.evaluate(test_X, test_y,verbose=0)

print("SPLIT1 Training completed in time: ", duration1)
print(f'SPLIT1 Validation accuracy {val_accuracy1[1]}')
print(f'SPLIT1 Test accuracy {test_accuracy1[1]}')

print("SPLIT2 Training completed in time: ", duration2)
print(f'SPLIT2 Validation accuracy {val_accuracy2[1]}')
print(f'SPLIT2 Test accuracy {test_accuracy2[1]}')

print("SPLIT3 Training completed in time: ", duration3)
print(f'SPLIT3 Validation accuracy {val_accuracy3[1]}')
print(f'SPLIT3 Test accuracy {test_accuracy3[1]}')
generate_plots(history2, 'Split3', test_X, test_y)

plt.show()
