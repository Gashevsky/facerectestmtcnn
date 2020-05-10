import cv2
import os
import threading
# develop a classifier for the 5 Celebrity Faces Dataset
from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
from os import listdir
from os.path import isdir
from numpy import asarray
from tensorflow.keras.models import load_model
from mtcnn.mtcnn import MTCNN
from PIL import Image

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras import backend as K
from tensorflow.compat.v1 import disable_eager_execution

# from tensorflow.python.client import device_lib 
# print(device_lib.list_local_devices())

K.clear_session()
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
disable_eager_execution()

class OutFrame:
    sharedFrame = None

# get the video and resize frame
def get_video(outFrame, lock, captureURL , captureMethod = cv2.CAP_FFMPEG):
	video_capture = cv2.VideoCapture(captureURL, captureMethod)
	video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)
	while True:
		# Capture frame-by-frame
		ret, frame = video_capture.read()	
		#budsize = cv2.get(cv2.CAP_PROP_BUFFERSIZE)
		# Display the resulting frame
		if ret:
			scale_percent = 25 # percent of original size
			width = int(frame.shape[1] * scale_percent / 100) 
			height = int(frame.shape[0] * scale_percent / 100) 
			dim = (width, height) 
			frameToShare = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA) 
			with lock:
				outFrame.sharedFrame = frameToShare

# load faces
# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
	X, y = list(), list()
	# enumerate folders, on per class
	for subdir in listdir(directory):
		# path
		path = directory + subdir + '/'
		# skip any files that might be in the dir
		if not isdir(path):
			continue
		# load all faces in the subdirectory
		faces = load_faces(path)
		# create labels
		labels = [subdir for _ in range(len(faces))]
		# summarize progress
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		# store
		X.extend(faces)
		y.extend(labels)
	return asarray(X), asarray(y)

# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
	# load image from file
	image = Image.open(filename)
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	pixels = asarray(image)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	# bug fix
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

def load_dataset_from_frame(frame):
	X = list()
	# enumerate folders, on per class	
	# load all faces in the subdirectory
	faces = extract_face_from_frame(frame)	
	if not(faces is None) and faces.any():
		X.extend(faces)	
	# store
	
	return asarray(X)

def extract_face_from_frame(frame, required_size=(160, 160)):
	# load image from file
	image = Image.fromarray(frame)
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	pixels = asarray(image)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	if results.__len__() == 0:
		return
	x1, y1, width, height = results[0]['box']
	# bug fix
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

# load images and extract faces for all images in a directory
def load_faces(directory):
	faces = list()
	# enumerate files
	for filename in listdir(directory):
		# path
		path = directory + filename
		# get face
		face = extract_face(path)
		# store
		faces.append(face)
	return faces

# get the face embedding for one face
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]


# load the facenet model
model = load_model('facenet_keras.h5')
print('Loaded Model')
# load face embeddings
data = load('faces-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
# normalize input vectors
in_encoder = Normalizer(norm='l2')
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy2 = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)
testX_faces = ''
#video_capture = cv2.VideoCapture(0)
#os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
#video_capture = cv2.VideoCapture("rtsp://admin:password@192.168.1.14:554",cv2.CAP_FFMPEG)
lock = threading.RLock()
sharedFrame = OutFrame()
frame = None
videograbThread = threading.Thread(target=get_video, args=(sharedFrame, lock,"rtsp://admin:Pass@192.168.1.14:554", cv2.CAP_FFMPEG), daemon=True)
videograbThread.start()
while True:
	if cv2.waitKey(1) & 0xFF == ord('q'):
			#video_capture.release()
			break	
	# Capture frame within lock
	with lock:
		frame = sharedFrame.sharedFrame 

	if not(frame is None):
		# Display the resulting frame
		testX_faces = load_dataset_from_frame(frame)
		cv2.imshow('Video', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		if testX_faces.any():
			# get embedding to the regognizing face
			newRecognizeX = list()
			embedding = get_embedding(model, testX_faces)
			newRecognizeX.append(embedding)
			newRecognizeX = asarray(newRecognizeX)

			trainX = in_encoder.transform(trainX)
			testX = in_encoder.transform(testX)
		
			# fit model
			model2 = SVC(kernel='linear', probability=True)
			model2.fit(trainX, trainy2)
			# recognize person
			selection = 0 
			random_face_pixels = testX_faces
			random_face_emb = newRecognizeX[selection]

			# prediction for the face
			samples = expand_dims(random_face_emb, axis=0)
			yhat_class = model2.predict(samples)
			yhat_prob = model2.predict_proba(samples)
			# get name
			class_index = yhat_class[0]
			class_probability = yhat_prob[0,class_index] * 100
			predict_names = out_encoder.inverse_transform(yhat_class)
			print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
			# plot face
			pyplot.imshow(random_face_pixels)
			title = '%s (%.3f)' % (predict_names[0], class_probability)
			pyplot.title(title)
			pyplot.show()
#video_capture.release()
cv2.destroyAllWindows()
model = None