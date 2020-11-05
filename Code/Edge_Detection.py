#Import libraries
import cv2
import numpy as np

#Read image.
img = cv2.imread('img1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gaussian = cv2.GaussianBlur(gray,(3,3),0)

#Resize imagage output
dsize = (440, 280)
output_image = cv2.resize(img, dsize, interpolation = cv2.INTER_AREA)

#Robert edge detection.
roberts_x = np.array([[0,0,0], [0,1,0], [0,0,-1]])
roberts_y = np.array([[0,0,0], [0,0,1], [0,-1,0]])
img_robertsx = cv2.filter2D(img_gaussian, -1, roberts_x)
img_robertsy = cv2.filter2D(img_gaussian, -1, roberts_y)
out_rob = np.square(img_robertsx) + np.square(img_robertsy)
output_roberts = cv2.resize(out_rob, dsize, interpolation = cv2.INTER_AREA)
#out = np.sqrt(np.square(img_robertsx) + np.square(img_robertsy))

#Sobel.
img_sobelx = cv2.Sobel(img_gaussian,cv2.CV_8U,1,0,ksize=5)
img_sobely = cv2.Sobel(img_gaussian,cv2.CV_8U,0,1,ksize=5)
out_sob = img_sobelx + img_sobely
img_sobel = cv2.resize(out_sob, dsize, interpolation = cv2.INTER_AREA)

#Prewitt.
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
out_pre = img_prewittx + img_prewitty
output_prewitty = cv2.resize(out_pre, dsize, interpolation = cv2.INTER_AREA)

#Canny.
im_cann = cv2.Canny(img,100,200)
img_canny = cv2.resize(im_cann, dsize, interpolation = cv2.INTER_AREA)

#Laplace of Gaussian
img_LoG = cv2.Laplacian(img_gaussian, cv2.CV_16S, ksize=3)
fin_LoG = cv2.convertScaleAbs(img_LoG)
final_LoG = cv2.resize(fin_LoG, dsize, interpolation = cv2.INTER_AREA)

cv2.imshow("Original Image", output_image)
cv2.imshow("Robert", output_roberts)
cv2.imshow("Sobel", img_sobel)
cv2.imshow("Prewitt", output_prewitty)
cv2.imshow("Canny", img_canny)
cv2.imshow("Laplace of Gaussian", final_LoG)

cv2.waitKey(0)
cv2.destroyAllWindows()