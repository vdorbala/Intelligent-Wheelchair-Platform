import cv2
import numpy as np

img = cv2.imread("./Images/test1/134.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
(kps, descs) = sift.detectAndCompute(gray, None)
cv2.drawKeypoints(gray, kps, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
height, width, channels = img.shape 
print ("Shape of image", height, width, channels)
print ("size of image:", np.size(gray))
print (np.size(kps),kps[1].pt[1])
k=0
for i in range(0,np.size(kps)):
	if ((kps[i].pt[0]) < (width/2)):
		# print (kps[i].pt[1])
		k = k+1

print ("Number on left is ", k)
print ("Number on right is ", np.size(kps)-k )
cv2.line(img,(width/2,0),(width/2,height),(0,255,0),2)

im = np.float32(img) / 255.0
 
gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)
hog = cv2.HOGDescriptor()
hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )

mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

cv2.imshow('SIFT Algorithm', im)
cv2.waitKey()
cv2.destroyAllWindows()