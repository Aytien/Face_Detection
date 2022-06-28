import os
import time
import uuid
import cv2

#select path for storing captured images
IMAGES_PATH =( os.path.join('data','train','images'),os.path.join('data','test','images'),os.path.join('data','val','images'))
number_images = (70,15,15)

#Capture Images with connected camera for 3 sets-trains,val,test
cap = cv2.VideoCapture(0)
for i in range(3):
    for imgnum in range(number_images[i]):
        print('Collecting image {}'.format(imgnum))
        ret, frame = cap.read()
        imgname = os.path.join(IMAGES_PATH[i],f'{str(uuid.uuid1())}.jpg')
        cv2.imwrite(imgname, frame)
        cv2.imshow('frame', frame)
        time.sleep(0.5)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

#Now label images with appropriate annotations and store in respective labels folder