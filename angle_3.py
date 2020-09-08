from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import imutils
import time
import math
import numpy as np
import cv2
from getperspective_1 import order_points,four_point_transform,warp_it

#read from IP camera
source = cv2.VideoCapture('rtsp://root:passw0rd@10.9.9.10/axis-media/media.amp?videocodec=h264&resolution=1024x768')
number=1
#Get the frame rate
frameRate = source.get(5)
#Take a snapshot of the videostream one second after the script is executed
while True:
    frameId = source.get(1) #current frame number
    _, frame = source.read()
    if ((frameId+5) % math.floor(1*frameRate) == 0):
        filename = 'C:\SharedFiles\Trays\image' +  str(int(number)) + ".jpg"
        cv2.imwrite(filename, frame)
        number+=1
    if number==2:
        break
#-------------------------------------------------------------------------------#
#In this section, we are trying to find the angle of the reference object

#Read in the snapshot that was taken 
img = cv2.imread('C:\SharedFiles\Trays\image1.jpg')
#Locate the reference object
frame_1 = img[300:350, 80:160]
#Blur the frame to reduce noise
blurred_frame=cv2.GaussianBlur(frame_1,(5,5),0)
#Define the color range of the reference object
lower_lock=np.array([100,10,150])
upper_lock=np.array([130,30,190])
#Convert the color type from BGR to HSV
hsv=cv2.cvtColor(blurred_frame,cv2.COLOR_BGR2HSV)
#Define a mask that is in the color range of the reference object
mask_reference=cv2.inRange(hsv,lower_lock,upper_lock)
#Create 'res' that only shows the mask
res= cv2.bitwise_and(frame_1,frame_1,mask=mask_reference)
#convert to grayscale
gray=cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)            
_,thresh=cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
#Find contours based on the threshold
contours, _=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

#Find the largest contour in the reference object frame
largest_area=0
largest_area_index=0
#find the largest contour 
for cnt in contours:
    area=cv2.contourArea(cnt)
    if area>largest_area :
        largest_area=area
        largest_contour_index=cnt
cv2.drawContours(res,largest_contour_index,-1,(255,255,0),2)
#------------------------------------------------------------------------#
#Find two points on the corners of the reference object (Point1: Point on the contour with the smallest sum value of coordinates / Point2: Point on the contour with the smallest difference value of coordinates)
summ_1=5000
coords_1=np.array([])
for i in range(len(largest_contour_index)):
    if sum(largest_contour_index[i][0]) < summ_1:
        summ_1=sum(largest_contour_index[i][0])
        coords_1=largest_contour_index[i][0]

diff_2=5000
coords_2=np.array([])
for i in range(len(largest_contour_index)):
    if  np.diff(largest_contour_index[i][0]) < diff_2:
        diff_2=np.diff(largest_contour_index[i][0])
        coords_2=largest_contour_index[i][0]
#Draw yellow circles of the points found above
cv2.circle(res, tuple(coords_1), 3, (0,255,255), 3)
cv2.circle(res, tuple(coords_2), 3, (0,255,255), 3)
#-------------------------------------------------------------------------#
#Find the angle of the reference object (with respect to the horizontal line)
angle_ref=(360/(2*math.pi))*(math.atan((coords_1[1]-coords_2[1])/(coords_2[0]-coords_1[0])))

#--------------------------------------------------------------------------#
#Now we find the angle of the pan

#Variables 
num=0
sticker_num=0
largest_area_2=0
largest_area_2_index=0

def angle_detect():
    #Test camera access 
    global num
    global largest_area_2
    global largest_area_2_index
    global largest_contour_index
    #select region of interest that shows only one edge of the pan
    img=frame[150:300, 270:550]
    blurred_frame=cv2.GaussianBlur(img,(5,5),0)
    #define color range of the pan edge
    lower_pan_edge=np.array([0,15,100])
    upper_pan_edge=np.array([50,100,255])
    
    #convert color range from BGR to HSV
    hsv=cv2.cvtColor(blurred_frame,cv2.COLOR_BGR2HSV)
    #create mask of the pan edge
    mask_pan=cv2.inRange(hsv,lower_pan_edge,upper_pan_edge)
    #res_1 only shows the pan
    res_1= cv2.bitwise_and(img,img,mask=mask_pan)
    #convert to grayscale
    gray=cv2.cvtColor(res_1,cv2.COLOR_BGR2GRAY)
    _,thresh=cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
    #draw contours based on the threshold
    contours, _=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    largest_area=0
    largest_area_index=0
    #find the largest contour 
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area>largest_area :
            largest_area=area
            largest_contour_index=cnt
    cv2.drawContours(res_1,largest_contour_index,-1,(255,255,0),2)
    #find two extreme points on the edge of the pan
    summ_1=5000
    coords_1=np.array([])
    for i in range(len(largest_contour_index)):
        if sum(largest_contour_index[i][0]) < summ_1:
            summ_1=sum(largest_contour_index[i][0])
            coords_1=largest_contour_index[i][0]

    diff_2=5000
    coords_2=np.array([])
    for i in range(len(largest_contour_index)):
        if  np.diff(largest_contour_index[i][0]) < diff_2:
            diff_2=np.diff(largest_contour_index[i][0])
            coords_2=largest_contour_index[i][0]
   
    cv2.circle(res_1, tuple(coords_1), 3, (0,255,255), 3)
    cv2.circle(res_1, tuple(coords_2), 3, (0,255,255), 3)
#------------------------------------------------------------#
# This section ensures that we can detect if the pan is turned 90 degrees. We are using the distance between two points to figure out
# whether the pan is turned 90 degrees from the "good orientation"
# Find the distance between two point found above    
    D = dist.euclidean((coords_1), (coords_2))
    #If distance is smaller than 220 print a statement
    if D < 220:
        print("The pan is in a very wrong orientation")
    
#Find the angle of the edge of the pan
    angle=(360/(2*math.pi))*(math.atan((coords_1[1]-coords_2[1])/(coords_2[0]-coords_1[0])))
    print('Rotate the pan  %2.2f degrees clockwise' %(-(angle_ref-angle)))

    cv2.imshow("res_1", res_1)
    cv2.imshow("gray", gray)


#---------------------------------------------------------------------------------#
#Below section takes a snapshot of the frame as soon as we see the pan in a smaller region of interest that we created

n=0
numb=0
while True:
    #reads the camera feed
    _,frame=source.read()
    #Define a smaller region of interest where we can only see on pan at a time
    frame_1 = frame[150:220, 200:800]
    blurred_frame=cv2.GaussianBlur(frame_1,(5,5),0)
    #Define the color range of the pan
    lower_pan=np.array([10,15,100])
    upper_pan=np.array([50,70,255])
    #Convert BGR to HSV
    hsv=cv2.cvtColor(blurred_frame,cv2.COLOR_BGR2HSV)

    #Create a mask for the pan
    mask_pan=cv2.inRange(hsv,lower_pan,upper_pan)
    #Create res which only shows the object in the color range of the pan
    res= cv2.bitwise_and(frame_1,frame_1,mask=mask_pan)
    #convert to grayscale
    gray=cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    _,thresh=cv2.threshold(gray,5,255,cv2.THRESH_BINARY)
    #find and draw contours of that frame
    contours, _=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(res,contours,-1,(0,255,0),2)
    largest_area=0
    largest_contour_index=0
    n=n+1
#-----------------------------------------------------------------------#
#Below section takes a snapshot when we see a pan in the smaller region of interest and feeds 
#this snapshot to the angle detect function
    try:
        if len(contours) > 0:
            
            for cnt in contours:
                area=cv2.contourArea(cnt)
                if area>largest_area :
                    largest_area=area
                    largest_contour_index=cnt

            if largest_area<(4000):
                n=0

            if largest_area>=4000 and n==1 :

                angle_detect()
                print(numb)
                numb+=1
 
                cv2.imwrite(filename="C:\SharedFiles\Trays1\Tray" + str(numb) + ".png", img=frame)
        cv2.imshow("res", res)
    except: 
        continue
    cv2.imshow("res", res)
    cv2.imshow("frame_1", frame_1)
    cv2.imshow("frame", frame)
    if cv2.waitKey(15) & 0xFF == ord('q'):
        break



source.release()
cv2.destroyAllWindows()