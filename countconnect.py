import torch
import cv2
import math
import time

model = torch.hub.load('/home/zidane/yolov5', 'custom', source='local', path='best2.pt', force_reload=True)
model.classes=[0]


cap = cv2.VideoCapture(0)
width = int(cap.get(3)); height = int(cap.get(4)); 

frameno=0
num_people=0
fpsStart = 0
fps = 0



# returns coordinates of box as list
def box_coords(box):
    xmin=int(box[0])
    ymin=int(box[1])
    xmax=int(box[2])
    ymax=int(box[3])
    return [xmin, ymin, xmax, ymax]

# checks if box touches the bottom of frame
def checkbot_box(coords,height):
    ymax=coords[3]
    if ymax>height-(height/54):
        return 1
    else:
        return 0

# returns center coordinates of box
def box_cent(coords):
    cent_x=int((coords[0]+coords[2])/2)
    cent_y=int((coords[1]+coords[3])/2)
    return [cent_x,cent_y]

# gets intersecting area of two boxes
def inters_area(coord1,coord2):
    xmin1=coord1[0]
    ymin1=coord1[1]
    xmax1=coord1[2]
    ymax1=coord1[3]
    xmin2=coord2[0]
    ymin2=coord2[1]
    xmax2=coord2[2]
    ymax2=coord2[3]
    dx=min(xmax1,xmax2)-max(xmin1,xmin2)
    dy=min(ymax1,ymax2)-max(ymin1,ymin2)
    if (dx>0) and (dy>0):
        return dx*dy
    else:
        return 0

# returns list of coordinates of boxes in current frame that are new (no corresponding box in previous frame)
def newbox(coordlist,i_list):
    new_list=[]
    for k in coordlist:
        if k not in [i[0] for i in i_list]:
            new_list+=[k]
    return new_list

# returns list of coordinates of boxes in previous frame that have disappeared (no corresponding box in current frame)
def dispbox(prev_coordlist,i_list):
    disp_list=[]
    for k in prev_coordlist:
        if k not in [i[1] for i in i_list]:
            disp_list+=[k]
    return disp_list

# finds which box in previous slide is the one in current frame (highest intersecting area)
def matchboxes(coordlist,prev_coordlist,width):
    i_list=[]
    for coord in coordlist:
        area=0
        add_ilist=[]
        for prev_coord in prev_coordlist:
            if inters_area(coord,prev_coord)>area and (math.dist(box_cent(coord),box_cent(prev_coord))<(4*width/20)):
                area=inters_area(coord,prev_coord)
                add_ilist=[[coord, prev_coord]]
            if coord not in [i[0] for i in i_list] and prev_coord not in [j[1] for j in i_list]:
                i_list+=add_ilist
    return i_list


# COUNT_PEOPLE_FRAMEOUT(prev_results, results, frame, rect_frame, num_people)
def COUNT_PEOPLE_FRAMEOUT(dataPre, dataCur, frame, frameCopy, num_people):
    # create lists of all box coordinates in previous and current frame
    prev_coordlist=[]
    for j in range(len(dataPre.xyxy[0])):
        prev_coords=box_coords(dataPre.xyxy[0][j])
        prev_coordlist+=[prev_coords]
    coordlist=[]
    for k in range(len(dataCur.xyxy[0])):
        coords=box_coords(dataCur.xyxy[0][k])
        coordlist+=[coords]
    
    for c in coordlist:
        cv2.rectangle(frameCopy,(c[0],c[1]),(c[2],c[3]),(255,0,0),thickness=-1)
    
    # list of boxes that have corresponding boxes in previous frame
    i_list=matchboxes(coordlist, prev_coordlist, width)
    
    # get list of boxes that are new in the frame
    new_list=newbox(coordlist,i_list)
    
    # get list of boxes that have disappeared
    disp_list=dispbox(prev_coordlist,i_list)
    
    # adjust number of people and draw rectangles
    for new_coords in new_list:
        if checkbot_box(new_coords,height)==1:
            num_people-=1
            cv2.rectangle(frameCopy,(new_coords[0],new_coords[1]),(new_coords[2],new_coords[3]),(0,0,255),thickness=-1)
    
    for disp_coords in disp_list:
        if checkbot_box(disp_coords,height)==1:
            num_people+=1
            cv2.rectangle(frameCopy,(disp_coords[0],disp_coords[1]),(disp_coords[2],disp_coords[3]),(0,255,0),thickness=-1)
    

    return frame, num_people




    
resultFINAL = cv2.VideoWriter('demovideo.avi', cv2.VideoWriter_fourcc(*'XVID'), cap.get(cv2.CAP_PROP_FPS), (width, height)) # 3 is FPS / cap.get(cv.CAP_PROP_FPS)

while(1):
    frameno+=1
    _, frame = cap.read()
    
    # create frames for color filling in
    rect_frame=frame.copy()


    results = model(frame)
    if frameno==1:
        prev_results=results
    


    frame, num_people = COUNT_PEOPLE_FRAMEOUT(prev_results, results, frame, rect_frame, num_people)


    fpsEnd = time.time()
    timeDiff = fpsEnd - fpsStart
    fps = 1/timeDiff
    fpsStart = fpsEnd

    fpsText  = "FPS: {:2.2f}".format(fps)
    cv2.putText(frame, fpsText, (int(width/40), int(height/15)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)    

    num_peopletxt="Number of people entered: "+str(num_people)
    if num_people>0:
        cv2.putText(frame, num_peopletxt, (int(width/40), height-int(width/40)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
    else:
        cv2.putText(frame, num_peopletxt, (int(width/40), height-int(width/40)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 0), 2)
    
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.imshow("result", frame)
    

    resultFINAL.write(frame)


    prev_results=results
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:

        break
    if k == 114 or k == 82:
        num_people = 0


cap.release()
resultFINAL.release()

cv2.destroyAllWindows()
