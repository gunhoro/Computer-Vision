from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import imutils
import time
import math
import numpy as np
import cv2
import glob
import random
from getperspective_1 import order_points,four_point_transform,warp_it
#-------------------------------------------------------------------------#
# Yolo detect function was obtained from  https://pysource.com/2019/07/08/yolo-real-time-detection-on-cpu/
# We first trained our images on google colab using google cloud gpu. '.weights' file is the result we get from the training. We use this file and the configuration file to 
#classify the images
#Read in the source
source = cv2.VideoCapture('rtsp://root:passw0rd@10.9.9.11/axis-media/media.amp?videocodec=h264&resolution=1024x768')
yolo = cv2.dnn.readNet("C:/darknet/cfg/yolov3_custom2_last_2.weights", "C:/darknet/cfg/yolov3_custom2_2.cfg")
classes=[]


with open("C:\darknet\data\line4.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = yolo.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in yolo.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 6))

colorRed = (0,0,255)
colorGreen = (0,255,0)


def yolo_detect():
    #Test camera access 
    img = cv2.imread("C:\SharedFiles\Train\\test\\test_" + str(numb) + ".jpg")
    height, width, channels = img.shape

    # # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    yolo.setInput(blob)
    outputs = yolo.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    #Find the bounding box with the greatest confidence
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = abs(int(center_x - w / 2))
                y = abs(int(center_y - h / 2))
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.2)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            a=x+w
            b=y+h
            if x+w > width:
                a=width-((x+w)-width)
                
            if y+h>height:
                b=height-((y+h)-height)
                
            cv2.rectangle(img, (x, y), (a, b), colorGreen, 3)
            cv2.putText(img, label, (x, y+30 ), cv2.FONT_HERSHEY_PLAIN, 2, colorRed, 2)
            
            if (len(confidences))>0:
                string=str(round(confidences[0]*100,2))
                cv2.putText(img, '%s%%' %(string), (x, y+100 ), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)
    
    if (len(confidences))>0:
        print('This is: %2.2f percent %s' %(confidences[0]*100, label))
    cv2.imshow("Image", img)
    cv2.imwrite("output.jpg",img)

#-----------------------------------------------------------------------------------------------------#
#Below section takes a snapshot as soon as we see the product in the region of interest. That snapshot is fed into the yolo_detect function 
n=0
numb=0
while True:
    #reads the camera feed
    _,frame=source.read()
    frame_1 = frame[900:1100, 220:650]
    frame_2 = frame[630:1200, 100:700]
    blurred_frame=cv2.GaussianBlur(frame_1,(5,5),0)
    lower_bag=np.array([0,100,100])
    upper_bag=np.array([30,255,255])
    
    hsv=cv2.cvtColor(blurred_frame,cv2.COLOR_BGR2HSV)
    
    mask_sticker=cv2.inRange(hsv,lower_bag,upper_bag)
    res= cv2.bitwise_and(frame_1,frame_1,mask=mask_sticker)
    gray=cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
        
    _,thresh=cv2.threshold(gray,5,255,cv2.THRESH_BINARY)
    contours, _=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    largest_area=0
    largest_contour_index=0
    n=n+1
    
    try:
        if len(contours) > 0:
            
            for cnt in contours:
                area=cv2.contourArea(cnt)
                if area>largest_area :
                    largest_area=area
                    largest_contour_index=cnt

            if largest_area<=(1400):
                n=0

            if largest_area>1400 and n==1 :   
                print(numb)
                numb+=1
                cv2.imwrite(filename="C:\SharedFiles\Train\\test\\test_" + str(numb) + ".jpg", img=frame_2)
                yolo_detect()
        
    except: 

        continue
    cv2.imshow("video", frame_2)
    if cv2.waitKey(15) & 0xFF == ord('q'):
        break
source.release()
cv2.destroyAllWindows()