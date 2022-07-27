import numpy as np
import cv2
import dlib
from scipy.spatial import distance
from playsound import playsound

import time
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    EAR = (A+B)/(2.0*C)
    return EAR

def lips_aspect_ratio(lips): 
    A = distance.euclidean(lips[4], lips[8])
    B = distance.euclidean(lips[2], lips[10])
    C = distance.euclidean(lips[0], lips[6])
    LAR = (A + B + C) / 3.0
    return LAR

# Declare another costant to hold the consecutive number of frames to consider for a blink 
CONSECUTIVE_FRAMES = 20 
# Initialize two counters 
BLINK_COUNT = 0 
frame_count = 0
count_sleep = 0

#######################################################################################

face_cascade = cv2.CascadeClassifier('haarcascades\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades\haarcascade_eye.xml')

img = cv2.imread('g1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#First detect face and then look for eyes inside the face.
#Multiscale refers to detecting objects (faces) at multiple scales. 
faces = face_cascade.detectMultiScale(gray, 1.3, 5) #scaleFactor = 1.3, minNeighbors = 3

for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)   #Draw red bounding box around the face
    roi_gray = gray[y:y+h, x:x+w] #Original gray image but only the detected face part
    roi_color = img[y:y+h, x:x+w] #Original color image but only the detected face part. For display purposes
    eyes = eye_cascade.detectMultiScale(roi_gray) #Use the gray face image to detect eyes
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2) #Draw green bounding boxes around the eyes

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

########################################################################################

#Apply the above logic to a live video
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascades\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades\haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor(r"E:\Mini Project\Drowsiness detection\proj4\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat")

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    #First detect face and then look for eyes inside the face.
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    
            facess = hog_face_detector(gray)    
            for face in facess:    
                face_landmarks = dlib_facelandmark(gray, face)
                leftEye = []
                rightEye = []
                for n in range(36,42):
                    x = face_landmarks.part(n).x
                    y = face_landmarks.part(n).y
                    leftEye.append((x,y))
                    next_point = n+1
                    if n == 41:
                        next_point = 36
                    x2 = face_landmarks.part(next_point).x
                    y2 = face_landmarks.part(next_point).y
                    cv2.line(img,(x,y),(x2,y2),(0,255,0),1)

                for n in range(42,48):
                    x = face_landmarks.part(n).x
                    y = face_landmarks.part(n).y
                    rightEye.append((x,y))
                    next_point = n+1
                    if n == 47:
                        next_point = 42
                    x2 = face_landmarks.part(next_point).x
                    y2 = face_landmarks.part(next_point).y
                    cv2.line(img,(x,y),(x2,y2),(0,255,0),1)

                left_ear = eye_aspect_ratio(leftEye)
                right_ear = eye_aspect_ratio(rightEye)

                EAR = (left_ear+right_ear)/2
                EAR = round(EAR,2)
                if EAR<0.26:
                    frame_count = frame_count + 1
                    if frame_count >= CONSECUTIVE_FRAMES:
                        count_sleep += 1
                        # Add the frame to the dataset ar a proof of drowsy driving
                        cv2.imwrite("dataset/frame_sleep%d.jpg" % count_sleep, img)
                        # playsound(r'C:\Users\Jeevan\OneDrive\Desktop\proj1\mixkit-facility-alarm-908.wav')
                        cv2.putText(img, "DROWSINESS ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        print("Drowsy")
                print(EAR)

            cv2.imshow("Are you Sleepy", img)
            key = cv2.waitKey(1)
            if key == 27:
                break
cap.release()
cv2.destroyAllWindows()