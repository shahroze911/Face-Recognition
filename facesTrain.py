import os
import cv2
from PIL import Image
from cv2 import face
import numpy as np
import pickle
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
image_dir=os.path.join(BASE_DIR,'Images')
face_cascade=cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eyes_cascade=cv2.CascadeClassifier('F:\PIAIC\Face Recognition\cascades\data\haarcascade_eye.xml')
smile_cascade=cv2.CascadeClassifier('F:\PIAIC\Face Recognition\cascades\data\haarcascade_smile.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()
current_id=0
label_ids={}
y_labels=[]
x_train=[]

for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith('jpg') or file.endswith('jpeg'):
            path=os.path.join(root,file)
            #label=os.path.basename(os.path.basename(path)).replace(" ","-").lower()
            #next two lines are same
            label=os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            #label=os.path.basename(root).replace(" ","-").lower()
            #print(label,path)
            if not label in label_ids:
                label_ids[label]=current_id
                current_id+=1
            id_=label_ids[label]
            #print(label_ids)
            #y_labels.append(label) #label
            #x_train.append(path) #verify this image and turn it into numpy array
            pil_image=Image.open(path).convert('L')
            #resize images
            size=(550,550)
            final_image=pil_image.resize(size,Image.ANTIALIAS)
            image_array=np.array(pil_image,"uint8")
            #print(image_array)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            for (x,y,w,h) in faces:
                roi=image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
            eyes=eyes_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            for (ex,ey,ew,eh) in eyes:
                roi=image_array[ey:ey+eh,ex:ex+ew]
                x_train.append(roi)
                y_labels.append(id_)
            smile = smile_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            for (sx, sy, sw, sh) in smile:
                roi = image_array[sy:sy + sh, sx:sx + sw]
                x_train.append(roi)
                y_labels.append(id_)

#print(y_labels)
#print(x_train)

with open("labels.pickle",'wb') as f:
    pickle.dump(label_ids,f)

recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainer.yml")




