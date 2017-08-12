
# coding: utf-8

# In[12]:

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

from keras.callbacks import TensorBoard, EarlyStopping

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from keras.utils import plot_model


# In[2]:

DATA_DIR = "/data/ymaab/data/PhotoAnyticRPRD/QFC/QD_4.v1/"
LOG_DIR = "log/"
num_epochs= 100


# In[3]:

base_model = ResNet50(weights='imagenet',include_top=False)


# In[8]:

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu', name='add_fc')(x)
# and a logistic layer -- let's say we have 4 classes
predictions = Dense(4, activation='softmax',name='last_sftmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)


# In[16]:

for layer in base_model.layers:
    layer.trainable = True

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])



# In[17]:

# plot_model(model, to_file='model.png', show_shapes=True)


# In[18]:

x_train = np.load(DATA_DIR+"qfc_x_train.npy")
y_train = np.load(DATA_DIR+"qfc_y_train.npy")
x_test = np.load(DATA_DIR+"qfc_x_test.npy")
y_test = np.load(DATA_DIR+"qfc_y_test.npy")


print x_test.max()
print x_test.min()
# In[23]:

x_train = x_train.astype('float32')
x_train = preprocess_input(x_train)


# In[24]:

# train the model on the new data for a few epochs
#model.fit(x_train, y_train,
#        batch_size=2,
#        epochs=100,
#        validation_split=0.1,
#        callbacks=[TensorBoard(log_dir=LOG_DIR+'/epoch_'+str(num_epochs)), EarlyStopping] )



model.fit(x_test, y_test,
        batch_size=2,
        epochs=100,
        validation_split=0.1)


# In[ ]:



