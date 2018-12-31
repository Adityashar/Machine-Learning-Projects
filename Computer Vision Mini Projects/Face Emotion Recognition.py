import cv2
import numpy as np

from keras.models import model_from_json
from keras.optimizers import Adam

json_file = open('model.json', 'r')
faces_model = json_file.read()
json_file.close()
faces_model = model_from_json(faces_model)
faces_model.load_weights('model.h5')

faces_model.compile(loss = "categorical_crossentropy", metrics = ['accuracy'], optimizer = Adam())

import pickle
x_train = []
file = open('faces_train.pkl', 'rb')
x_train = pickle.load(file)

from sklearn.preprocessing import MinMaxScaler
data = x_train.reshape(-1, 2304)
scale = MinMaxScaler(feature_range = (-1,1))
data = scale.fit_transform(data)
x_train = data.reshape(-1, 48, 48, 1)
cv2.imshow("", x_train[0])
cv2.waitKey()
cv2.destroyAllWindows()

emotions = ['angry', 'disgust', 'scared', 'happy', 'sad', 'surprised', 'neutral']

def Find_label(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48), interpolation = cv2.INTER_AREA)    
   # print("Resized !!", face.shape)
    
    face = face.reshape(-1, 2304)
    face = scale.transform(face)
   # print("Transformed !!")
    
    face = face.reshape(-1, 48, 48, 1)
    label = faces_model.predict(face)
    
    return np.argmax(label)

face_classifer = cv2.CascadeClassifier('E:\OpenCV\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')

def face_detector(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray_image, (3, 3), 0)
    
    faces = face_classifer.detectMultiScale(gray_blur, 1.2, 2)       # SCALING FACTOR AND NUMBER OF NEIGHBOURS WITHIN A RECTANGLE
    if faces is () :
        return image
    
    for (x, y, w, h) in faces:
        crop_image = image[y:y+h, x:x+w]
        predicted_label = Find_label(crop_image)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(image, str(emotions[predicted_label]), (x+w-70, y+h-10), cv2.FONT_HERSHEY_COMPLEX, 1, (125, 127, 0), 1)
    
    #image = cv2.flip(image, 1)
    return image

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow("Face", face_detector(frame))
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()