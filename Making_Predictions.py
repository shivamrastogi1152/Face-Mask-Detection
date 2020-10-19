import tensorflow
from keras.models import load_model
import cv2
import numpy as np


model = load_model('./final_model')

face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)

labels_dict={1:'MASK',0:'NO MASK'}
color_dict={1:(0,255,0),0:(0,0,255)}


while(True):

    ret,img=cap.read()
    # img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.flip(img,1)
    faces=face_clsfr.detectMultiScale(img,1.3,5)  

    for (x,y,w,h) in faces:
    
        face_img=img[y:y+h,x:x+w]
        resized=cv2.resize(face_img,(96,96))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,96,96,3))
        result=model.predict(reshaped)

        label=np.argmax(result,axis=1)[0]
        probability = result[0][label]
        accuracy = probability*100;
        accuracy = round(accuracy,2)
      
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        cv2.putText(img, str(accuracy)+str("%"), (x+w-100, y-10),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),2)
            
            
    cv2.imshow('LIVE',img)    
    key=cv2.waitKey(1) & 0xFF
    if(key==ord('q')):
        break
    
    
cap.release()        
cv2.destroyAllWindows()



