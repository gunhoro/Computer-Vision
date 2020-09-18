# Computer-Vision
4 files in this repository are the computer vision projects completed during the summer internship at Martin's Famous Pastry Shoppe

2.11.2.py : Rejects trays that contains objects that are undesired. Extensive image processing techniques were used. Communication with the PLC server through openOPC. Time in between trays in the tray line is calculated. Details are in the comments of the script. 


Product_identification.py : Identifies which product passes by in one of the lines using YOLO algorithm. 

angle_3.py : Computes the angle of the pan that moves along the pan line. This information would later be used in fixing the angle of improperly oriented pans. 

line4_2.py: Rejects products that are improperly bagged(tightly or loosely bagged, no product in the bag, no clips, etc) using a blower. Communication with the PLC server through openOPC.
