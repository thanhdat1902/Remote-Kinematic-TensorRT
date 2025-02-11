import cv2
import os
vidcap = cv2.VideoCapture('./Video/p24_front_1.mp4')
success,image = vidcap.read()
count = 0

start=0
name="P24"
frame = 0

start_frame = 0

while success:
    success,image = vidcap.read()
    if success:
        if frame == start_frame:
            path = os.path.join(os.getcwd(), "{}/frame_{:06d}.PNG".format(name, start))
            cv2.imwrite(path, image)     # save frame as JPEG file      
            print('Read a new frame: ', path, success)
            count += 1
            start+=1
            if count == 1001:
                break
        else:
            frame+=1
        