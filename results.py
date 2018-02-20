import numpy as np
import cv2
import pandas as pd

result = np.array(pd.read_csv("prediction.csv"))
i = int(input("Input index :"))
i = result[i]
image = cv2.imread("test/"+i[1])
image = cv2.resize(image,(100,100))
cv2.imshow(i[0],image)
cv2.waitKey(0)
print(i)
#print(type(image))
