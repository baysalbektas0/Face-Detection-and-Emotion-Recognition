import cv2
import os
from keras.models import load_model
from config import *
from utils import get_yolo_boxes
from keras.preprocessing.image import img_to_array
import numpy as np
import time
import utils


cv2.namedWindow('Emotion_Detection')
camera = cv2.VideoCapture(0)
model = load_model("./weights/shufflenetv2.h5")
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]
emotion_model_path = 'emotiondetection.h5'
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry","disgust","fear", "happy", "neutral", "sad", "surprise"]
while True:
    stime = time.time()
    width  = int(camera.get(3))# float
    height = int(camera.get(4))
    ret, frame = camera.read()
    image = cv2.resize(frame,(width,height))
    frameClone = image.copy()
    
    
    if ret:
        boxes = get_yolo_boxes(model, [image], net_w, net_h, anchors, obj_thresh, nms_thresh)[0]
        
        for box in boxes:
            xmin=(box.xmin)
            ymin=(box.ymin)
            xmax=(box.xmax)
            ymax=(box.ymax)
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = gray[ymin:ymax,xmin:xmax]
        gray = cv2.resize(gray, (48,48))
        gray = gray.astype("float") / 255.0
        gray = img_to_array(gray)
        gray = np.expand_dims(gray, axis=0)
       
        
        preds = emotion_classifier.predict(gray)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
        
            
    text2 = "{}: {:.2f}%".format(label, emotion_probability * 100)        
    cv2.putText(frameClone, text2, (xmin, ymin - 10),
    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    cv2.rectangle(frameClone, (xmin,ymin), (xmax,ymax),
                  (0, 0, 255), 2) 
        
    cv2.imshow('Emotion_Detection', frameClone)

    print('FPS {:.1f}'.format(1 / (time.time() - stime)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()