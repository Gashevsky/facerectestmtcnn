from datetime import datetime as dt
from datetime import timedelta as tdelta
import cv2
import threading
import requests
import pyodbc 

from numpy import expand_dims
from matplotlib import pyplot
from numpy import asarray
from tensorflow.keras.models import load_model
from mtcnn.mtcnn import MTCNN
from PIL import Image
import joblib

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
			scale_percent = 50 # percent of original size
			width = int(frame.shape[1] * scale_percent / 100) 
			height = int(frame.shape[0] * scale_percent / 100) 
			dim = (width, height) 
			frameToShare = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA) 
			with lock:
				outFrame.sharedFrame = frameToShare

def load_dataset_from_frame(frame, detector):
	facesList = list()
	# enumerate folders, on per class	
	# load all faces in the subdirectory
	faces = extract_face_from_frame(frame, detector)	
	if not(faces is None) and faces.any():
		facesList.extend(faces)	
	# store	
	return asarray(facesList)

def extract_face_from_frame(frame, detector, required_size=(160, 160)):
	# load image from file
	image = Image.fromarray(frame)
	# convert to RGB, if needed
	#image = image.convert('RGB')
	# convert to array
	pixels = asarray(image)	
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	if results.__len__() == 0:
		return
	x1, y1, width, height = results[0]['box']
	# we do not recognize wery small picture
	if height < 160 or width < 160:
		return 
	x1, y1 = abs(x1), abs(y1)
	xCorrection = int(0)
	yCorrection = int(0)
	(frameHeight, frameWidth, frameDepth) = frame.shape
	if width > height:
		yCorrection = (width - height) // 2
		height = width
		if (y1 - yCorrection < 0) or (y1 + yCorrection > frameHeight):
			return # the face is too close to the border
		y1 = y1 - yCorrection

	if width < height:
		xCorrection = (height - width) // 2
		height = width
		if (x1 - xCorrection < 0) or (x1 + xCorrection > frameWidth):
			return # the face is too close to the border
		x1 = x1 - xCorrection
		
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

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

# send command to ESP relay controller
def openDoor():	
	urlGet = 'https://192.168.1.84/command/open'
	myobj = "{\"pass\":\"P_Test\"}"
	#x = requests.get(url, data = myobj)
	getResult = requests.post(urlGet, data = myobj, verify = False)

# create the face detector, using default weights
detectorMTCNN = MTCNN()
# load the facenet model
modelFaceNetPretrained = load_model('facenet_keras.h5')
print('Loaded Model')

# # load face embeddings
# data = load('faces-embeddings.npz')
# trainX, trainy = data['arr_0'], data['arr_1']
# # normalize input vectors
# in_encoder = Normalizer(norm='l2')
# # label encode targets
# out_encoder = LabelEncoder()
# out_encoder.fit(trainy)
# trainy2 = out_encoder.transform(trainy)
# #load model
# model2 = SVC(kernel='linear', probability=True)
# trainX = in_encoder.transform(trainX)
# # fit model			
# model2.fit(trainX, trainy2)

(modelSVC_trained, out_encoder) = joblib.load("pickle_svc_model.joblib")

detectedFacesArray = ''
#video_capture = cv2.VideoCapture(0)
#os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
#video_capture = cv2.VideoCapture("rtsp://admin:password@192.168.1.14:554",cv2.CAP_FFMPEG)
lock = threading.RLock()
sharedFrame = OutFrame()
frame = None

recognitionTime = dt.today() # uses to calculate delta to prevent open door repeat 
repeatOpenCommandAfter = tdelta(seconds = 15)
videograbThread = threading.Thread(target=get_video, args=(sharedFrame, lock,"rtsp://admin:Test1234@192.168.1.15:554", cv2.CAP_FFMPEG), daemon=True)
#videograbThread = threading.Thread(target=get_video, args=(sharedFrame, lock,"rtsp://admin:@192.168.1.77:554/h264/ch1/main/av_stream", cv2.CAP_FFMPEG), daemon=True)
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
		detectedFacesArray = load_dataset_from_frame(frame, detectorMTCNN)
		cv2.imshow('Video', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		if detectedFacesArray.any():			
			newRecognizeX = list()
			# get embedding to the regognizing face
			embedding = get_embedding(modelFaceNetPretrained, detectedFacesArray)
			newRecognizeX.append(embedding)
			newRecognizeX = asarray(newRecognizeX)							
			# recognize person
			selection = 0 
			captured_face_pixels = detectedFacesArray
			captured_face_emb = newRecognizeX[selection]

			# prediction for the face
			samples = expand_dims(captured_face_emb, axis=0)		
			predicted_class = modelSVC_trained.predict(samples)
			predicted_probability = modelSVC_trained.predict_proba(samples)
			# get name
			class_index = predicted_class[0]
			class_probability = predicted_probability[0,class_index] * 100
			predict_names = out_encoder.inverse_transform(predicted_class)
			print('Predicted: %s (%.6f)' % (predict_names[0], class_probability))
			if (class_probability > 99.999):
				conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=DESKTestWIN10HO;DATABASE=OpenDoors;UID=Gashevskyi;PWD=Test1234')
				cursor = conn.cursor()
				rows = cursor.execute("select UserId from Users where UserName = ?", predict_names[0]).fetchall()
				userId = 0
				if ((rows is not None) and len(rows) == 1):
					userId = rows[0].UserId
					_,encodedPng = cv2.imencode(".png",frame)
					cursor.execute("insert into OpenDoorLogs (UserId, ImageData, RecognitionProbability) values (?,?,?)",(2, pyodbc.Binary(encodedPng), class_probability))
					conn.commit()

				if (class_probability > 99.99999): #send command to controller 
					curTime = dt.today()
					if (curTime > (recognitionTime + repeatOpenCommandAfter)):
						openDoor()
						recognitionTime = dt.today()
						# plot face
						pyplot.imshow(captured_face_pixels)
						title = '%s (%.6f)' % (predict_names[0], class_probability)
						pyplot.title(title)
						pyplot.show()
				conn.close()
#video_capture.release()
cv2.destroyAllWindows()
modelFaceNetPretrained = None
