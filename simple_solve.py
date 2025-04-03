from PIL import Image
import scipy
import glob
import numpy as np
import scipy.ndimage
import cv2
import skimage
#from skimage import img_as_ubyte
import matplotlib.pyplot as plt

im=Image.open('images/pieces_black.jpg') #Requires a black background

gray_im = im.convert('L')
gray_im = np.asarray(gray_im,dtype="uint8")/255.0

binarized_image=np.zeros(gray_im.shape)

binarized_image[gray_im>0.2]=1.0


image=scipy.ndimage.binary_opening(binarized_image,
                 structure=np.ones((5,5)), iterations=1, output=None, origin=0)
image=scipy.ndimage.binary_closing(image,
             structure=np.ones((5,5)), iterations=1, output=None, origin=0)

#image[image>0.06]=1.0
cv_image = skimage.util.img_as_ubyte(image)
contours, hierarchy = cv2.findContours(cv_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
im2 = cv_image.copy()  # Create a copy for drawing
im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2BGR)

#perimeter = cv2.arcLength(cnt,True)
cv2.drawContours(im2,contours,-1,(255,255,0),2)
plt.imshow(im2,cmap=plt.cm.gray)
plt.title("Pieces")
plt.axis('off')
plt.show()

#recon_img =np.zeros((h1,w1))
piece_array = []
for i in range(0,len(contours)):
        c =np.asarray(contours[i])
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(im2,(x,y),(x+w,y+h),(0,255,0),2)
        crop_img = gray_im[y:y+h, x:x+w] # Crop from x, y, w, h -> 100, 200, 300, 400

        #crop_img=np.asarray(crop_img,dtype="uint8")
        piece_array.append(crop_img)

plt.imshow(im2,cmap=plt.cm.gray)
plt.axis('off')
plt.show()

# Show the first 3 extracted pieces in a new plot
num_pieces = min(3, len(piece_array))  # Ensure we don't exceed available pieces

plt.figure(figsize=(10, 5))  # Create a new figure

for i in range(num_pieces):
    plt.subplot(1, 3, i + 1)  # Create 1 row, 3 columns of subplots
    plt.imshow(piece_array[i], cmap="gray")
    plt.title(f"Piece {i+1}")
    plt.axis("off")

plt.show()

