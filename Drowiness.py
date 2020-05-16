#importing all important and necessary libraries
import cv2
import dlib
from playsound import playsound
from scipy.spatial import distance as dist
import time


cap=cv2.VideoCapture(0)
count=0 #to count the number of frames
face_detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
time.sleep(1.0)
while cap.isOpened():
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_detector(gray) # to detect the face
    for face in faces:
        cv2.rectangle(frame,(face.left(),face.top()),(face.right(),face.bottom()),(0,0,255),3)
        landmarks=predictor(gray,face) # to detect the landmarks
        for n in range(36,48):
           cv2.circle(frame, (landmarks.part(n).x,landmarks.part(n).y), 3, (0, 255, 0), -1)
            #to calculate the ledt eye's EAR
        p=[(landmarks.part(36).x, landmarks.part(36).y),(landmarks.part(37).x,landmarks.part(37).y),
           (landmarks.part(38).x,landmarks.part(38).y),(landmarks.part(39).x,landmarks.part(39).y),
           (landmarks.part(40).x,landmarks.part(40).y),(landmarks.part(41).x,landmarks.part(41).y)]
        dis1=dist.euclidean(p[0],p[3])
        dis2 = dist.euclidean(p[1], p[5])
        dis3= dist.euclidean(p[2], p[4])
        L_EAR=(dis2+dis3)/(2*dis1)
        print(L_EAR)
        # to calculate the right eye's EAR
        r = [(landmarks.part(42).x, landmarks.part(42).y), (landmarks.part(43).x, landmarks.part(43).y),
             (landmarks.part(44).x, landmarks.part(44).y), (landmarks.part(45).x, landmarks.part(45).y),
             (landmarks.part(46).x, landmarks.part(46).y), (landmarks.part(47).x, landmarks.part(47).y)]
        dis1_1 = dist.euclidean(r[0], r[3])
        dis2_2 = dist.euclidean(r[1], r[5])
        dis3_3 = dist.euclidean(r[2], r[4])
        R_EAR = (dis2_2+ dis3_3) / (2 * dis1_1)
        EAR=(L_EAR+R_EAR)/2 #EAR is calculated
        print(EAR)
        cv2.putText(frame, "EAR: {:.2f}".format(EAR), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if EAR<0.22:
            count=count+1
            if count>=4:
                cv2.putText(frame, "Drowsiness_Detected", (100,130),
                        cv2.FONT_HERSHEY_SIMPLEX,1, (255,0, 0), 2)
            if count>=5:
                playsound('alarm.wav') # sound is played.
        elif EAR>=0.22:
            count=0
    cv2.imshow("FRAME",frame)
    if cv2.waitKey(1) ==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()