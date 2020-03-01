# example of loading the keras facenet model
from tensorflow.keras.models import load_model
# load the model
model = load_model('facenet_keras.h5')
# summarize input and output shape
print(model.inputs)
print(model.outputs)


# confirm mtcnn was installed correctly
import mtcnn
# print version
print(mtcnn.__version__)