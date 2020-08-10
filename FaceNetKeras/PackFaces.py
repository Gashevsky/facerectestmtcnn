# inspired by https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/
import tensorflow.keras
import concurrent.futures
from os import listdir
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
from numpy import expand_dims
from mtcnn.mtcnn import MTCNN
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from numpy import load
import joblib
import pyodbc 

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.models import load_model
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
	pictures, labels = list(), list()
	importdirs = list(listdir(directory))
	# enumerate folders, on per class
	with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
	#with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
		future_to_data = {executor.submit(load_directory, directory, subdir): subdir for subdir in importdirs}
		for future in concurrent.futures.as_completed(future_to_data):
			subdir = future_to_data[future]
			try:
				faces, face_labels = future.result()
				print('>loaded files for subdir:', subdir) 
				if ((faces is not None) and (labels is not None)):
					pictures.extend(faces)
					labels.extend(face_labels)
			except Exception as exc:
				print('%r generated an exception: %s' % (subdir, exc))
			else:
				print('%r page is %d bytes' % (subdir, len(faces)))
	#for subdir in importdirs:		
		#faces, labels = load_directory(directory, subdir)		
	return asarray(pictures), asarray(labels)

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

conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=DESKTestWIN10HO;DATABASE=OpenDoors;UID=Gashevskyi;PWD=Test1234')
cursor = conn.cursor()

# load test dataset
trainPhotos, trainLabels = load_dataset('faces-dataset/')
# save arrays to one file in compressed format
savez_compressed('faces-dataset.npz', trainPhotos, trainLabels)

#data = load('faces-dataset.npz')
#trainPhotos, trainLabels = data['arr_0'], data['arr_1']
print('Loaded: ', trainPhotos.shape, trainLabels.shape)
# load the facenet model
model = load_model('facenet_keras.h5')
print('Loaded Model')
# convert each face in the train set to an embedding
trainPicturesEmbedding = list()
for face_pixels in trainPhotos:
	embedding = get_embedding(model, face_pixels)
	trainPicturesEmbedding.append(embedding)
trainPictures = asarray(trainPicturesEmbedding)
# save arrays to one file in compressed format
savez_compressed('faces-embeddings.npz', trainPicturesEmbedding, trainLabels)
#data = load('faces-embeddings.npz')

print(trainPicturesEmbedding)
for trainLabel in trainLabels:
	rows = cursor.execute("select * from Users where UserName = ?", trainLabel).fetchall()
	if (rows is None) or len(rows) == 0:
		cursor.execute("insert into Users (UserName) values (?)", trainLabel)
		conn.commit()
inp_normalizer = Normalizer(norm='l2')
label_encoder = LabelEncoder()
label_encoder.fit(trainLabels)
modelSVC = SVC(kernel='linear', probability=True)
trainPictures = inp_normalizer.transform(trainPictures)
transformedLabels = label_encoder.transform(trainLabels)
# fit model			
modelSVC.fit(trainPictures, transformedLabels)

svc_model_filename = "pickle_svc_model.joblib"
model_encoder = (modelSVC, label_encoder)
joblib.dump(model_encoder, svc_model_filename)