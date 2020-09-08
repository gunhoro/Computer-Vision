from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import imutils
import time
import math
import numpy as np
import cv2
import glob
from numpy import linalg as LA
import random
from getperspective_1 import order_points,four_point_transform,warp_it
import xlsxwriter
import OpenOPC
from Blower import Blow
import OpenOPC
import time
#opc = OpenOPC.open_client('172.22.250.29')
#opc.connect('RSLinx OPC Server')
#opc.list('PatternMaker4B')
#opc = OpenOPC.open_client('172.22.250.16')
#opc.connect('RSLinx OPC Server')
#var = opc.list('PatternMaker4b.online.VisionBadProductBlowOff')
 
#------------------------------------------------------------------------------------------#
#Function line4 analyzes the snapshot of a product. It tells us if a product is properly bagged or not. 
source = cv2.VideoCapture('rtsp://root:passw0rd@10.9.9.11/axis-media/media.amp?videocodec=h264&resolution=1024x768')
workbook=xlsxwriter.Workbook('C:\SharedFiles\GunHo\Final\\bag_data_1.xlsx')
worksheet = workbook.add_worksheet()
improper_count=0
def line4():
    global improper_count
    img=cv2.imread("C:\SharedFiles2\Master_Versions\Test1\\test_" + str(numb) + ".jpg")
    
    #############################################################
    #Drawing a thick line to cover up the side parts of line4. 
    side_start=(600,6)
    side_end=(570,447)
    #separate the side
    cv2.line(img, side_start, side_end, (0,0,0), 100) 
    #######################################################
    #Finding contours entire product including the bag selecting the color range of the belt color(blue). By using
    #bitwise_not you exclude all color that is blue in the frame which leaves the product. 
    blurred_frame1=cv2.GaussianBlur(img,(15,15),0)
    lower_blue=np.array([100,180,130])
    upper_blue=np.array([115,255,255])
    lower_side=np.array([10,30,130])
    upper_side=np.array([50,80,225])
    hsv=cv2.cvtColor(blurred_frame1,cv2.COLOR_BGR2HSV)
    mask_blue=cv2.inRange(hsv,lower_blue,upper_blue)
    mask_side=cv2.inRange(hsv,lower_side,upper_side)
    mask=cv2.bitwise_or(mask_blue,mask_side)
    mask=cv2.bitwise_not(mask)
    res1= cv2.bitwise_and(img,img,mask=mask)

    gray1=cv2.cvtColor(res1,cv2.COLOR_BGR2GRAY) 
    _,thresh=cv2.threshold(gray1,5,255,cv2.THRESH_TOZERO)
    contours, _=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) 
    #Find the largest contour 
    largest_contour_index3=0
    largest_area3=0
    for cnt in contours:
            area=cv2.contourArea(cnt)
            if area>largest_area3 :
                largest_area3=area
                largest_contour_index3=cnt
    cv2.drawContours(img,largest_contour_index3,-1,(255,255,0),2)
    #######################################################
    #In this secion, we find a line that bisects the entire contour of the product
    rows,cols = img.shape[:2]
    [vx,vy,x,y] = cv2.fitLine(largest_contour_index3, cv2.DIST_L2,0,0.01,0.01)
    lefty = int((-x*vy/vx) + y)
    righty = int(((cols-x)*vy/vx)+y)
    cv2.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)
    total_area=cv2.contourArea(largest_contour_index3)
    
    ##############################################################
    #In this part, we aim to find the location of the clip using the function convexHull
    #We find the points that form the convex shape from the entire product contour
    hull=cv2.convexHull(largest_contour_index3,returnPoints=False)
    #Using the points from convexHull function, convexityDefects returns the start and end points of the convex line, the point
    #on the product contour that is farthest from the convex line and the distance between the line and that point. 
    defects = cv2.convexityDefects(largest_contour_index3,hull)
    color = (255, 255, 255)
    howfar=0
    how_far_list=[]
    #Then we find two points on the contour that is farthest from the convex lines. We can predict that the 
    #center of those two points would be where the clip is. 
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        how_far_list.append(d)
    far_index=sorted(range(len(how_far_list)), key=lambda k: how_far_list[k],reverse=True)
    s1,e1,f1,d1=defects[far_index[0],0]
    start1 = tuple(largest_contour_index3[s1][0])
    end1 = tuple(largest_contour_index3[e1][0])
    far1 = tuple(largest_contour_index3[f1][0])
    s2,e2,f2,d2=defects[far_index[1],0]
    start2 = tuple(largest_contour_index3[s2][0])
    end2 = tuple(largest_contour_index3[e2][0])
    far2 = tuple(largest_contour_index3[f2][0])
    
    cv2.line(img,start1,end1,[0,255,0],2)
    cv2.line(img,start2,end2,[0,255,0],2)
    far_1=(far1[0]-1000,((1000*vx/vy)+far1[1])[0])
    cv2.fillPoly(gray1, pts =[largest_contour_index3], color=(255,255,255))
    #cv2.line(gray1,far_1,(far1[0]+1000,((-1000*vx/vy)+far1[1])[0]),[0,0,255],6)
    #clip_point
    clip=(int((far1[0]+far2[0])/2),int((far1[1]+far2[1])/2))
    clip_1=(clip[0]-1000,((1000*vx/vy)+clip[1])[0])
    ###########################################################
    #dividing the tail and body of the bag
    #Draw a line that is perpendicular to the line of fit and that goes through the clip point
    cv2.line(gray1,clip_1,(clip[0]+1000,((-1000*vx/vy)+clip[1])[0]),[0,0,255],3)
    cv2.line(img,clip_1,(clip[0]+1000,((-1000*vx/vy)+clip[1])[0]),[0,0,255],3)
    _,thresh_d=cv2.threshold(gray1,200,255,cv2.THRESH_TOZERO)
    contours, _=cv2.findContours(thresh_d,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)  

    largest_area=0
    largest_area_index=0
    #find the largest contour 
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area>largest_area :
            largest_area=area
            largest_contour_index=cnt
    cv2.drawContours(img,largest_contour_index,-1,(255,255,255),2)
    M=cv2.moments(largest_contour_index)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
    body_area=cv2.contourArea(largest_contour_index)
    tail_area=total_area-body_area
    print('body area is :%f / tail_area is: %f /total_area is: %f'  %(body_area,tail_area,total_area))
    ##############################################
    #Find an arrow that points to the tail
    #We do so by comparing the distance between each intersection points and the clip point. The one closer to the clip point 
    #should be the tail side. 
    cv2.arrowedLine(img, (cX,cY), tuple(clip), (255,0,0), 2)
    if cX>clip[0]:
        print('Wrong Direction')
    cv2.circle(img,far1,3,[0,0,255],-1)
    cv2.circle(img,far2,3,[0,0,255],-1)
    cv2.circle(img,clip,3,[0,0,255],-1)
    #####################################################################
    #Computing different parameters such as distance between one end and another end, left end to clip, right end to clip, 
    #distance between the convex line to the clip, ratio of tail area and body area etc 
    
    far1tofar2=dist.euclidean(far1,far2)
    bodycenter2clip=dist.euclidean(clip,(cX,cY))
    print('bodycenter2clip: %f' %(bodycenter2clip))
    p1=np.asarray(start1)
    p2=np.asarray(end1)
    p3=np.asarray(clip)
    clip2convex=LA.norm(np.cross(p2-p1, p1-p3))/LA.norm(p2-p1)
    print('distance between clip and convex line :%f ' %clip2convex)
    cv2.imshow('Frame',img)
    # worksheet.write(numb,0,numb)
    # worksheet.write(numb,1,body_area)
    # worksheet.write(numb,2,tail_area)
    # worksheet.write(numb,3,total_area)
    # worksheet.write(numb,4,ratio)
    # worksheet.write(numb,5,clip2convex)
    # worksheet.write(numb,6,end2end)
    # threshold_min_area=20000
    # threshold_max_area=30000
    #threshold for body area
    threshold_min_area=18000
    threshold_max_area=23000
    #threshold for convex length
    threshold_min_length=100
    #threshold for clip to center of body area
    body2clip_min_length=70
    body2clip_max_length=140

    if body_area < threshold_min_area  or body_area>threshold_max_area or far1tofar2>threshold_min_length or cX>clip[0] or bodycenter2clip> body2clip_max_length or \
        bodycenter2clip < body2clip_min_length :     
        improper_count+=1
        cv2.imwrite(filename="C:\SharedFiles2\Master_Versions\Test1\Bad1\Bad_" + str(numb) + ".jpg", img=frame_2)
        #Blow()
        if improper_count>=3:
            print('%d consecutive improper products' %(improper_count))
    if body_area < threshold_min_area  or body_area>threshold_max_area:
        print('body area problem')
        cv2.imwrite(filename="C:\SharedFiles2\Master_Versions\Test1\BodyArea\Bad_" + str(numb) + ".jpg", img=frame_2)
    if far1tofar2>threshold_min_length:
        print('convex to convex length problem')
        cv2.imwrite(filename="C:\SharedFiles2\Master_Versions\Test1\Convex\Bad_" + str(numb) + ".jpg", img=frame_2)
    if cX>clip[0]:
        print('direction problem')
        cv2.imwrite(filename="C:\SharedFiles2\Master_Versions\Test1\Direction\Bad_" + str(numb) + ".jpg", img=frame_2)
    if bodycenter2clip> body2clip_max_length or bodycenter2clip < body2clip_min_length :
        print('body to clip problem')
        cv2.imwrite(filename="C:\SharedFiles2\Master_Versions\Test1\BodyClip\Bad_" + str(numb) + ".jpg", img=frame_2)
    if body_area > threshold_min_area  and body_area<threshold_max_area and far1tofar2<threshold_min_length and cX<clip[0] and bodycenter2clip< body2clip_max_length and \
        bodycenter2clip > body2clip_min_length:
        improper_count=0

#-------------------------------------------------------------------------------------------#
#Below section takes a snapshot whenever the blue belt background is obstructed by the product 
#in the region of interest. It feeds that snapshot into line4 function. 
n=0
numb=0
scale_percent=60
while True:
    _,frame=source.read()
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("frame", frame)
    frame_1 = frame[400:450, 200:300]
    frame_2 = frame[200:500, 100:390]
    blurred_frame=cv2.GaussianBlur(frame_1,(5,5),0)
    lower_blue=np.array([100,150,200])
    upper_blue=np.array([120,255,255])
    hsv=cv2.cvtColor(blurred_frame,cv2.COLOR_BGR2HSV)
    mask_sticker=cv2.inRange(hsv,lower_blue,upper_blue)
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
  
            if largest_area>=(1500):
                n=0
            if largest_area<1500 and n==1 :
                print(numb)
                numb+=1
                cv2.imwrite(filename="C:\SharedFiles2\Master_Versions\Test1\\test_" + str(numb) + ".jpg", img=frame_2)
                line4()
                
        cv2.imshow("res", res)
        
    except: 
        continue

    cv2.imshow("res", res)
    cv2.imshow("frame", frame)
    if cv2.waitKey(15) & 0xFF == ord('q'):
        break

#workbook.close()

source.release()
cv2.destroyAllWindows()