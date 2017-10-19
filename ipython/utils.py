import numpy as np
import itertools
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import confusion_matrix

import threading


def getlabel(arr):
	maxvalue=-1
	maxp=0
	for idx, val in enumerate(arr):
		if val > maxvalue:
			maxvalue = val
			maxp = idx
	return maxp

def plot_confusion_matrix(cm, classes,
						  normalize=False,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

def cropcenter(img):
	
	width = img.shape[1]
	height = img.shape[0]
	length = min(width, height)
#	 print("width:%d"%width)
#	 print("height:%d"%height)



	left = (width - length)/2
#	 print("left:%d"%left)
	top = (height - length)/2
#	 print("top:%d"%top)
	right = (width + length)/2
#	 print("right:%d"%right)
	bottom = (height + length)/2
#	 print("bottom:%d"%bottom)
	
	

#	 img = img[top:bottom, left:right,::]

	im = Image.fromarray(np.uint8(img))
#	 print(im.size)
#	 print(left,top,right,bottom)
	im = im.crop((left,top,right,bottom))
#	 print(im.size)

	im = im.resize((224,224))
	
	img = np.array(im)
	
	return img


class threadsafe_iter(object):
  """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
  def __init__(self, it):
      self.it = it
      self.lock = threading.Lock()

  def __iter__(self):
      return self

  def __next__(self):
      with self.lock:
          return self.it.__next__()

def threadsafe_generator(f):
  """
    A decorator that takes a generator function and makes it thread-safe.
    """
  def g(*a, **kw):
      return threadsafe_iter(f(*a, **kw))
  return g