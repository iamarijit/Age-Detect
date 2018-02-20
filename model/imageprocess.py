import numpy as np
import cv2

#imgList contains the names of image files
#imgDir is the path where the images are located
def imgprocess(imgList,imgDir):
    images = []
    for img in imgList:
        #reading image
        image = cv2.imread(imgDir+img)
        #scaling image
        image = cv2.resize(image,(28,28))
        #image = image.astype('float32')
        #image = 255
        images.append(image)

    images = np.array(images)
    return images
