"""
PROJECT VISION 20
Martins Famous Pastry Shoppe Inc.
-----------------------------------------
Version 1.11.2 S.C.
Designed for Stale Line B61 Reject
-----------------------------------------
Written by Gun Ho Ro and Drew Martin Summer 2020

With Help From Kyle Myers, Nichole Wahl, Derek Piper, Alex Brennecke, Tony Martin, Jose Maita, Adrian Rosenbrock, and Many Others
Thank you for all of your help! This woudn't be possbile without you!
-----------------------------------------
BEGIN SCRIPT 1.11.2 S.C.
"""
#Import All Libraries
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import imutils
import time
import math
import numpy as np
import cv2
from Lights2 import Red, Green
import OpenOPC
from getperspective_1 import order_points,four_point_transform,warp_it
from datetime import datetime

#source = cv2.VideoCapture('rtsp://root:passw0rd@10.9.1.30/axis-media/media.amp?videocodec=h264&resolution=1024x768')
# number=1
# frameRate = source.get(5)
# while True:
#     frameId = source.get(1) #current frame number
#     _, frame = source.read()
#     if ((frameId+5) % math.floor(1*frameRate) == 0):
#         filename = 'C:\SharedFiles2\scale_photo\image' +  str(int(number)) + ".jpg"
#         cv2.imwrite(filename, frame)
#         number+=1
#     if number==2:
#         break
# print('done')
    

# def midpoint(ptA, ptB):
# 	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
# PixelsPerInch=36/3
# img = cv2.imread('C:\SharedFiles2\scale_photo\image1.jpg')


# # cv2.imshow('image',img)
# # k = cv2.waitKey(0)
# # if k == 27:         # wait for ESC key to exit
# #     cv2.destroyAllWindows()


# frame_1 = img[420:555, 630:710]
# blurred_frame=cv2.GaussianBlur(frame_1,(5,5),0)
# lower_yellow=np.array([65,80,150])
# upper_yellow=np.array([85,115,200])

# hsv=cv2.cvtColor(blurred_frame,cv2.COLOR_BGR2HSV)

# mask_sticker=cv2.inRange(hsv,lower_yellow,upper_yellow)
# res= cv2.bitwise_and(frame_1,frame_1,mask=mask_sticker)
# gray=cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
# edged = cv2.Canny(gray, 50, 100)
# edged = cv2.dilate(edged, None, iterations=1)
# edged = cv2.erode(edged, None, iterations=1)
    

# cnts =cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
# cnts = imutils.grab_contours(cnts)
# box = cv2.minAreaRect(max(cnts, key = cv2.contourArea))
# box = cv2.boxPoints(box)
# box = np.array(box, dtype="int")
# box = perspective.order_points(box)
# (tl, tr, br, bl) = box
# (tltrX, tltrY) = midpoint(tl, tr)
# (blbrX, blbrY) = midpoint(bl, br)

# (tlblX, tlblY) = midpoint(tl, bl)
# (trbrX, trbrY) = midpoint(tr, br)
# dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
# dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
# dimA = dA / PixelsPerInch
# dimB = dB / PixelsPerInch
# print(dA)
# print(dimA)
# scale_factor=dimA/5.71
# print('scale factor: %f' %(scale_factor))
# scale = input("Scale Factor? y/n: ")
# #---------------------------------------------------------------------#
# if scale == ("n" or "N"):
#     scale_factor=1
scale_factor=1
#---------------------------------------------------------------------#
opc = OpenOPC.open_client('172.22.250.29')
opc.connect('RSLinx OPC Server')
record = input("Record y/n: ")
#Variables
num=0
sticker_num=0
largest_area_2=0
largest_area_2_index=0
#Camera IP Address
source = cv2.VideoCapture('rtsp://root:passw0rd@10.9.1.30/axis-media/media.amp?videocodec=h264&resolution=1024x768')
#var = opc.list('Stale1Chbg.online.program:RemotePanel_5')
#print (var)
#Color Detection Function
def color_detect():
    global num
    global largest_area_2
    global largest_area_2_index
    print(numb)
    img=cv2.imread("C:\SharedFiles2\\allTrays4\\tray" + str(numb) + ".png")
    #img=frame_2
    #Blur the frame to get rid of noise
    blurred_frame=cv2.GaussianBlur(img,(5,5),0)
    #define color range
    lower_blue=np.array([100,80,150])
    upper_blue=np.array([121,255,255])
    #convert color range from BGR to HSV for processing
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #create specific mask around basket
    mask_basket=cv2.inRange(hsv,lower_blue,upper_blue)
    #looks at mask instead of entire image
    res= cv2.bitwise_and(img,img,mask=mask_basket)
    #make res grayscale
    gray=cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    #set threshold
    _,thresh=cv2.threshold(gray,5,255,cv2.THRESH_BINARY)
    #draw contours based on the threshold
    contours, _=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #pre-define variables to make python happy
    largest_area=0
    largest_area_index=0
    #find the largest contour 
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area>largest_area :
            largest_area=area
            largest_contour_index=cnt
    summ_3=0
    #define four corners of tray based on contours
    coords_3=np.array([])
    for i in range(len(largest_contour_index)):
        if sum(largest_contour_index[i][0]) >= summ_3:
            summ_3=sum(largest_contour_index[i][0])
            coords_3=largest_contour_index[i][0]
    summ_1=5000
    coords_1=np.array([])
    for i in range(len(largest_contour_index)):
        if sum(largest_contour_index[i][0]) < summ_1:
            summ_1=sum(largest_contour_index[i][0])
            coords_1=largest_contour_index[i][0]
    diff_4=-5000
    coords_4=np.array([])
    for i in range(len(largest_contour_index)):
        if  np.diff(largest_contour_index[i][0]) > diff_4:
            diff_4=np.diff(largest_contour_index[i][0])
            coords_4=largest_contour_index[i][0]
    diff_2=5000
    coords_2=np.array([])
    for i in range(len(largest_contour_index)):
        if  np.diff(largest_contour_index[i][0]) < diff_2:
            diff_2=np.diff(largest_contour_index[i][0])
            coords_2=largest_contour_index[i][0]
    #draw circles on those cornerse
    cv2.circle(img, tuple(coords_3), 3, (0,255,255), 3)
    cv2.circle(img, tuple(coords_1), 3, (0,255,255), 3)
    cv2.circle(img, tuple(coords_4), 3, (0,255,255), 3)
    cv2.circle(img, tuple(coords_2), 3, (0,255,255), 3)
    pts=np.array([tuple(coords_1),tuple(coords_2),tuple(coords_3),tuple(coords_4)])
    #correct the warped image with perspective transform
    warped = four_point_transform(img, pts)
    height,width=warped.shape[:2]
    a=0.20
    b=0.92
    c=0.05
    d=0.95
    e=(b-a)/2
    f=(d-c)/2
    #defined the four sections of the tray
    warped_all = warped[int(height*a):int(height*b),int(width*c):int(width*d)]
    warped_1 = warped[int(height*a):int(height*(a+e)),int(width*c):int(width*(c+f))]
    warped_2 = warped[int(height*a):int(height*(a+e)),int(width*(c+f)):int(width*d)]
    warped_3 = warped[int(height*(a+e)):int(height*b),int(width*c):int(width*(c+f))]
    warped_4 = warped[int(height*(a+e)):int(height*b),int(width*(c+f)):int(width*d)]
    #create a quick and slick array to hold the warped image values
    warped_list=[warped_1,warped_2,warped_3,warped_4]
    #Blur some more frames to eliminate noise
    area_check=np.array([])
    blurred_frame_warped=cv2.GaussianBlur(warped_1,(5,5),0)
    #change BGR to HSV 
    hsv=cv2.cvtColor(blurred_frame_warped,cv2.COLOR_BGR2HSV)
    #create a mask for any unknown objects in the region(s) of interest
    mask_object=cv2.inRange(hsv,lower_blue,upper_blue)
    #Only look at the mask
    res_warped= cv2.bitwise_and(warped_1,warped_1,mask=mask_object)
    #grayscale that image
    gray_warped=cv2.cvtColor(res_warped,cv2.COLOR_BGR2GRAY)
    #create a threshold
    _,thresh_warped=cv2.threshold(gray_warped,105,255,cv2.THRESH_BINARY)
    #find contours based on that threshold
    contours, _=cv2.findContours(thresh_warped,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #set the value of that threshold
    threshold_area=70 * (scale_factor**2)
    k=0
    #determine if the tray coming through is dirty OR its an older model with wider holes
    for cnt in contours:
        area2=cv2.contourArea(cnt)
        if area2>threshold_area*scale_factor**2 and area2<15000*scale_factor**2:
            print(area2)
            area_check=np.append(area_check, area2)
    #print(area_check)
    if len(area_check) > 5:
        k=1
        print('Older Tray With Wider Holes')
    else:
        k=0
    #find contour areas and set their values to variables    
    area_check2=np.array([])
    if k==0:
        lower_blue=np.array([90,30,180])
        upper_blue=np.array([121,255,255])
        thres_min=140*scale_factor**2
        thres_max=13000*scale_factor**2
        thres_all_min=140*scale_factor**2
        thres_all_max=55000*scale_factor**2
    elif k==1:
        a=0.20
        b=0.92
        c=0.12
        d=0.92
        e=(b-a)/2
        f=(d-c)/2
        warped_all = warped[int(height*a):int(height*b),int(width*c):int(width*d)]
        warped_1 = warped[int(height*a):int(height*(a+e)),int(width*c):int(width*(c+f))]
        warped_2 = warped[int(height*a):int(height*(a+e)),int(width*(c+f)):int(width*d)]
        warped_3 = warped[int(height*(a+e)):int(height*b),int(width*c):int(width*(c+f))]
        warped_4 = warped[int(height*(a+e)):int(height*b),int(width*(c+f)):int(width*d)]
        warped_list=[warped_1,warped_2,warped_3,warped_4]
        lower_blue=np.array([100,60,150])
        upper_blue=np.array([121,255,255])
        thres_min=400*scale_factor**2
        thres_max=10000*scale_factor**2
        thres_all_min=400*scale_factor**2
        thres_all_max=50000*scale_factor**2
    
    blurred_frame_all=cv2.GaussianBlur(warped_all,(5,5),0)
    hsv=cv2.cvtColor(warped_all,cv2.COLOR_BGR2HSV)
    mask_object=cv2.inRange(hsv,lower_blue,upper_blue)

    res_warped= cv2.bitwise_and(warped_all,warped_all,mask=mask_object)
    gray_warped=cv2.cvtColor(res_warped,cv2.COLOR_BGR2GRAY)
    _,thresh_warped=cv2.threshold(gray_warped,105,255,cv2.THRESH_BINARY)
    contours, _=cv2.findContours(thresh_warped,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    
    for cnt in contours:
        area2=cv2.contourArea(cnt)

        
        if area2>thres_all_min*scale_factor**2 and area2 <thres_all_max*scale_factor**2  :
            print(area2)
            area_check2=np.append(area_check2, area2)
    
    for i in warped_list:
            
        blurred_frame_warped=cv2.GaussianBlur(i,(5,5),0)
        hsv=cv2.cvtColor(blurred_frame_warped,cv2.COLOR_BGR2HSV)
        mask_object=cv2.inRange(hsv,lower_blue,upper_blue)
        res_warped= cv2.bitwise_and(i,i,mask=mask_object)
        gray_warped=cv2.cvtColor(res_warped,cv2.COLOR_BGR2GRAY)
        _,thresh_warped=cv2.threshold(gray_warped,105,255,cv2.THRESH_BINARY)
        contours, _=cv2.findContours(thresh_warped,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(i,contours,-1,(0,255,0),2)
        for cnt in contours:
            area2=cv2.contourArea(cnt)
            if area2>thres_min*scale_factor**2 and area2 < thres_max*scale_factor**2 :
                print(area2)
                area_check2=np.append(area_check2, area2)
    #If the areas are = 0: Turn On The Green Light 
    #GOOD        
    if len(area_check2)==0:
        #This references the Lights.py file which essentially just blinks the light of choice with OpenOPC
        Green()
    #If the areas are larger than 0: Turn On The Red Light
    #BAD
    if len(area_check2)>0:
        num += 1
        print('THIS TRAY GOT DIRRRRRT')
        #This references the Lights.py file which essentially just blinks the light of choice with OpenOPC
        Red()
        #If you choose to record at the beginning of the script, it will save all dirty trays to this file path~~
        if record == ("y" or "Y"):
            cv2.imwrite(filename="C:\SharedFiles2\\DirtyTrays4\\tray" + str(num) + ".png", img=frame_2)
    #Optional show all four sections seperately 
        """
    cv2.imshow("roi", warped ) 
    cv2.imshow("warped_1", warped_1) 
    cv2.imshow("warped_2", warped_2) 
    cv2.imshow("warped_3", warped_3) 
    cv2.imshow("warped_4", warped_4) 
    #cv2.imshow("warped_sticker", res_warped_sticker) 
"""
n=0
numb=0
time_list=[0,0]
current_time_list=[]
t=list(range(1,100))
u=np.ones([1,100])
#print(u)
count=0
#MAIN LOOP
while True:
    #reads the camera feed
    _,frame=source.read()
    #Specific dimensions
    frame_1 = frame[170:400, 430:550]
    frame_2 = frame[160:590, 280:710]
    #Blur frame to eliminate noise
    blurred_frame=cv2.GaussianBlur(frame_1,(5,5),0)
    #set HSV Color Range of tape/sticker
    lower_orange=np.array([70,85,170])
    upper_orange=np.array([100,130,255])
    #Change the frame from BGR to HSV
    hsv=cv2.cvtColor(blurred_frame,cv2.COLOR_BGR2HSV)
    #make mask for the sticker/tape
    mask_sticker=cv2.inRange(hsv,lower_orange,upper_orange)
    #only look at that range
    res= cv2.bitwise_and(frame_1,frame_1,mask=mask_sticker)
    #grayscale the frame image
    gray=cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    #create threshold
    _,thresh=cv2.threshold(gray,5,255,cv2.THRESH_BINARY)
    #draw contours based on the threshold
    contours, _=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #pre-define some variables to make python happy
    largest_area=0
    largest_contour_index=0
    n=n+1
    #find the largest contour 
    
    try:
        if len(contours) > 0:
            for cnt in contours:
                area=cv2.contourArea(cnt)
                #print(area)
                if area>largest_area :
                    largest_area=area
                    largest_contour_index=cnt
            if largest_area>=(600)*scale_factor**2:
                n=0
            if largest_area<600*scale_factor**2 and n==1 :
                count=0
                u=np.ones([1,100])
                #run main function
                photo_time=time.time()
                #print(photo_time)
                cv2.imwrite(filename="C:\SharedFiles2\\allTrays4\\tray" + str(numb) + ".png", img=frame_2)
                color_detect()
                numb+=1
                if (numb % 2) == 0:
                    time_list[0]=photo_time
                elif (numb %2 ) == 1:
                    time_list[1]=photo_time
                
                interval=abs(time_list[0]-time_list[1])
                #print(interval)
                if interval > 120:
                    print('we have not seen a tray for more than %.2f minutes' %(interval/60))
                    now=datetime.now()
                    current_time=now.strftime("%H:%M:%S")
                    print("Current TIme =" ,current_time)
                    current_time_list=np.append(current_time_list,current_time)
                    print(current_time_list)
                #write all pictures for debugging
                
                #if choose to record at beginning of script: save all tray images to this file path~~
                if record == ("Y" or "y"):
                    cv2.imwrite(filename="C:\SharedFiles\Tray" + str(numb) + ".png", img=frame_2)
            if (numb%2) == 0:
                time_since_lasttray=time.time()-time_list[0]
                
            elif (numb%2) == 1:
                time_since_lasttray=time.time()-time_list[1]
            #print(t[count])
            #print(time_since_lasttray)
            if time_since_lasttray<30000000 and time_since_lasttray>3*t[count] and u[0][count]==1:
                print('no tray for %f seconds' %(3*t[count]))
                u[0][count]+=1
                count+=1
                
            
            

    except: 
        continue
    #option to show different images in different windows
    #cv2.imshow("res", res)
    cv2.imshow("video", frame)
    cv2.imshow("Video", frame_2)
    #Set the delay between Loops
    if cv2.waitKey(1) & 0xFF == ord('q'):
    #Check for the letter q to quit ^^^
        break
source.release()
cv2.destroyAllWindows()