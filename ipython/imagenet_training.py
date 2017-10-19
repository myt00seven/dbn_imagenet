
# coding: utf-8

# In[19]:


from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

from keras.callbacks import TensorBoard, EarlyStopping
from keras.callbacks import ModelCheckpoint, CSVLogger

# import sys
# sys.path.inser(0,'')
from keras.applications.resnet50  import ResNet50
from keras.applications.resnet50  import preprocess_input, decode_predictions

# from myresnet50 import ResNet50
# from myresnet50 import preprocess_input, decode_predictions

import time
import numpy as np
from keras.utils import plot_model

# import pickle
import hickle as hkl
import glob,os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import threading

# from utils import getlabel, plot_confusion_matrix, cropcenter, threadsafe_generator


LOG_DIR = "log/"
num_epochs= 100


DATA_MAIN_PATH = '/mnt/yma/imagenet-hkl/'
LABELS_PATH = DATA_MAIN_PATH+'labels/'
TRAIN_FOLDER = 'train_hkl_b256_b_256/'
# TRAIN_FOLDER = 'val_hkl_b256_b_256/'
TEST_FOLDER = 'val_hkl_b256_b_256/'
HKL_EMBED_FIGURE = 256
LOAD_LENGTH=32
TRAIN_PATH = DATA_MAIN_PATH + TRAIN_FOLDER
TEST_PATH = DATA_MAIN_PATH + TEST_FOLDER

STEPS_PER_EPOCH_TRAIN = 5003*HKL_EMBED_FIGURE/LOAD_LENGTH/10
# STEPS_PER_EPOCH_TRAIN = 10
STEPS_PER_EPOCH_VAL   = 194 *HKL_EMBED_FIGURE/LOAD_LENGTH/10
# STEPS_PER_EPOCH_VAL   = 1

# GIT_DATA_PATH = '/home/lab.analytics.northwestern.edu/yma/git/data/'
GIT_DATA_PATH = '../../git/data/'

set_type = 'train'
model = 'resnet50_rminit_bn'
NUM_CLASS = 1000

weights_file = ""


def class_to_array(y_class):
    batch_size = len(y_class)
    arr = np.zeros(shape=(batch_size,NUM_CLASS))
    for i in range(batch_size):
        arr[i][y_class[i]]=1
    return arr   




class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator 
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
                        # print("Batch: %s, for %s"%(file, set_type))
                        hkl_file = hkl.load(file_path)


                        X = np.swapaxes(hkl_file,0,3)
                        X = X.astype("float32")

                        # X[..., 0] -= 103.939
                        # X[..., 1] -= 116.779
                        # X[..., 2] -= 123.68

                        X = X / 255.0
                        X -= 0.5
                        X *= 2.


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
csv_logger = CSVLogger(GIT_DATA_PATH+'cvslogs/' + model + '-' + 'training-' +     str(timestamp) + '.log')
# model = ResNet50(weights=None,include_top=True)

model = ResNet50(weights='imagenet',include_top=True)
# model = ResNet50(weights=None,include_top=True)

print("Finish generating model.")

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
print("Finish compiling model.")

# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])

if weights_file != "":
    print("Loading saved model: %s." % weights_file)
    model.load_weights(weights_file, by_name=True)


callbacks = [checkpointer, tb, early_stopper, csv_logger]


print model.metrics_names

# res = model.evaluate_generator(
#         imagenet_loader('val'),
#         steps=190)

# print res

train_generator = imagenet_loader('train')
val_generator   = imagenet_loader('val')

model.fit_generator(
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH_TRAIN,
        validation_data=val_generator,
        validation_steps=STEPS_PER_EPOCH_VAL,
        epochs=num_epochs,
        callbacks=callbacks,
        workers = 2,
        max_queue_size = 200
        )




