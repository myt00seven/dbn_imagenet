
# coding: utf-8

# In[19]:


from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

from keras.callbacks import TensorBoard, EarlyStopping
from keras.callbacks import ModelCheckpoint, CSVLogger

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions

import time
import numpy as np
from keras.utils import plot_model

# import pickle
import hickle as hkl
import glob,os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# In[20]:


LOG_DIR = "log/"
num_epochs= 100


DATA_MAIN_PATH = '/mnt/imagenet-hkl/'
LABELS_PATH = DATA_MAIN_PATH+'labels/'
# TRAIN_FOLDER = 'train_hkl_b256_b_256/'
TRAIN_FOLDER = 'val_hkl_b256_b_256/'
TEST_FOLDER = 'val_hkl_b256_b_256/'
HKL_EMBED_FIGURE = 256
LOAD_LENGTH=32
TRAIN_PATH = DATA_MAIN_PATH + TRAIN_FOLDER
TEST_PATH = DATA_MAIN_PATH + TEST_FOLDER

# GIT_DATA_PATH = '/home/lab.analytics.northwestern.edu/yma/git/data/'
GIT_DATA_PATH = '../../git/data/'


# In[21]:


set_type = 'train'
model = 'resnet50_rminit'
NUM_CLASS = 1000

# In[ ]:

def class_to_array(y_class):
    batch_size = len(y_class)
    arr = np.zeros(shape=(batch_size,NUM_CLASS))
    for i in range(batch_size):
        arr[i][y_class[i]]=1
    return arr   

def imagenet_loader(set_type):
    if set_type == 'train':
        HKL_PATH = TRAIN_PATH
        label = np.load(LABELS_PATH+'train_labels.npy')
    elif set_type == 'test' or set_type == 'val':
        HKL_PATH = TEST_PATH
        label = np.load(LABELS_PATH+'val_labels.npy')

    start = HKL_EMBED_FIGURE
    length = LOAD_LENGTH
    end = HKL_EMBED_FIGURE

    while True:

        if start == end:

            for root, dirs, files in os.walk(HKL_PATH):
                for file in files:
                    if file.endswith(".hkl"):
                        file_path = os.path.join(root, file)
                        # print("Extracting a batch from: %s, for %s"%(file_path, set_type))
                        print("Batch: %s, for %s"%(file, set_type))
                        hkl_file = hkl.load(file_path)
                        X = np.swapaxes(hkl_file,0,3)
                        if X.shape[1]==256 and X.shape[2]==256:
                            X = X[::,16:240, 16:240, ::]

                        numbers = [int(s) for s in file.split('.') if s.isdigit()]
                        batch_index = numbers[0]
                        label_start = batch_index * HKL_EMBED_FIGURE
                        label_end   = label_start + HKL_EMBED_FIGURE
                        
                        y_class = label[label_start:label_end]
                        y= class_to_array(y_class)

                        # print(X.shape)

                        start = 0
                        yield (X[start:start+LOAD_LENGTH], y[start:start+LOAD_LENGTH])
        else:
            start = start+LOAD_LENGTH
            yield (X[start-LOAD_LENGTH:start], y[start-LOAD_LENGTH:start])


# In[22]:


# Helper: Save the model.
checkpointer = ModelCheckpoint(
        filepath=GIT_DATA_PATH+'checkpoints/' + model + '-' + set_type + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5',
        verbose=1,
        save_best_only=True)

# Helper: TensorBoard
tb = TensorBoard(log_dir= GIT_DATA_PATH +'logs/')

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(patience=10)

# Helper: Save results.
timestamp = time.time()
csv_logger = CSVLogger(GIT_DATA_PATH+'logs/' + model + '-' + 'training-' +     str(timestamp) + '.log')


# In[23]:


model = ResNet50(weights=None,include_top=True)

print("Finish generating model.")

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
print("Finish compiling model.")

# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])



callbacks = [checkpointer, tb, early_stopper, csv_logger]


# In[ ]:


model.fit_generator(
        imagenet_loader('train'),
        steps_per_epoch=100,
        validation_data=imagenet_loader('val'),
        validation_steps=2,
        epochs=num_epochs,
        callbacks=callbacks)


# In[ ]:




