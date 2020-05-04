# face detection for the 5 Celebrity Faces Dataset https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/
import tensorflow.keras
import concurrent.futures
from os import listdir
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True

session = InteractiveSession(config=config)
tensorflow.compat.v1.disable_eager_execution()
# extract a single face from a given photograph
def extract_face(filename, detector, required_size=(160, 160)):
	# load image from file
	print('extracting face from: %s' % (filename))
	try:
		image = Image.open(filename)
		# convert to RGB, if needed
		image = image.convert('RGB')
		# convert to array
		pixels = asarray(image)		
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
	except Exception as exc:
		print('%r .open generated an exception: %s' % (filename, exc))
		return None
	return face_array

# load images and extract faces for all images in a directory
def load_faces(directory):
	faces = list()
	# create the detector, using default weights
	detector = MTCNN()
	# enumerate files	
	for filename in list(listdir(directory)):
		# path
		path = directory + filename	
		# get face
		face = extract_face(path, detector)
		# store
		if (face is not None):
			faces.append(face)
	return faces

def load_directory(directory, subdir):
		# path
		path = directory + subdir + '/'
		# skip any files that might be in the dir
		if not isdir(path):
			return None, None
		# load all faces in the subdirectory
		faces = load_faces(path)
		# create labels
		labels = [subdir for _ in range(len(faces))]
		# summarize progress
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		# store
		return faces, labels

# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
	X, Y = list(), list()
	importdirs = list(listdir(directory))
	# enumerate folders, on per class
	with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
	#with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
		future_to_data = {executor.submit(load_directory, directory, subdir): subdir for subdir in importdirs}
		for future in concurrent.futures.as_completed(future_to_data):
			subdir = future_to_data[future]
			try:
				faces, labels = future.result()
				print('>loaded files for subdir:', subdir) 
				if ((faces is not None) and (labels is not None)):
					X.extend(faces)
					Y.extend(labels)
			except Exception as exc:
				print('%r generated an exception: %s' % (subdir, exc))
			else:
				print('%r page is %d bytes' % (subdir, len(faces)))
	#for subdir in importdirs:		
		#faces, labels = load_directory(directory, subdir)		
	return asarray(X), asarray(Y)

# load train dataset
trainX, trainy = load_dataset('faces-dataset/train/')
print(trainX.shape, trainy.shape)
# load test dataset
testX, testy = load_dataset('faces-dataset/val/')
# save arrays to one file in compressed format
savez_compressed('faces-dataset.npz', trainX, trainy, testX, testy)