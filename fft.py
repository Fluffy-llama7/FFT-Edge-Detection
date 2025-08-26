import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread("image.jpg", 0)
#image output is matrix
discrete_ft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
#shift zero frequency to center of array
d_shift = np.fft.fftshift(discrete_ft)
m_spectrum = 20 * np.log(cv2.magnitude(d_shift[:,:,0], d_shift[:,:,1]))
#magnitude spectrum - low frequencies get blocked
#circular high pass filter mask

#make mask
rows, cols = img.shape
crow, ccol = int(rows/2), int(cols/2) #center coordinates

#mask 1 for edge detection
mask = np.ones((rows, cols, 2), np.uint8)
r = 80
center = [crow, ccol]
x,y = np.ogrid[:rows, :cols]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
mask[mask_area] = 0

#mask 2 for blurring
mask2 = np.zeros((rows, cols, 2), np.uint8) #zero matrix
r = 80 #radius
#center = [crow, ccol]
#x,y = np.ogrid[:rows, :cols]
#mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
mask2[mask_area] = 1

#apply mask and inverse DFT
fshift = d_shift * mask #fft transformed image values multiplied by mask values
f_ishift = np.fft.ifftshift(fshift) #ifftshift - inverse fft
img_back = cv2.idft(f_ishift) #idft - inverse dft
#inverse dft converts back from frequency to image domain
img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

#apply mask 2 and inverse DFT
fshift2 = d_shift * mask2 #fft transformed image values multiplied by mask values
f_ishift2 = np.fft.ifftshift(fshift2) #ifftshift - inverse fft
img_back2 = cv2.idft(f_ishift2) #idft - inverse dft
#inverse dft converts back from frequency to image domain
img_back2 = cv2.magnitude(img_back2[:,:,0], img_back2[:,:,1])

#band pass filter  
r_out, r_in = 120, 30
mask3 = np.zeros((rows, cols, 2), np.uint8)
center = (crow, ccol)
x, y = np.ogrid[:rows, :cols]
distance = (x - crow)**2 + (y - ccol)**2
mask3[(distance >= r_in**2) & (distance <= r_out**2)] = 1

#apply mask 3
fshift3 = d_shift * mask3 #fft transformed image values multiplied by mask values
f_ishift3 = np.fft.ifftshift(fshift3) #ifftshift - inverse fft
img_back3 = cv2.idft(f_ishift3) #idft - inverse dft
#inverse dft converts back from frequency to image domain
img_back3 = cv2.magnitude(img_back3[:,:,0], img_back3[:,:,1])


fig = plt.figure(figsize=(15, 10))

ax1 = fig.add_subplot(2,3,1)
ax1.imshow(img, cmap='gray')
ax1.title.set_text('Input Image')

ax2 = fig.add_subplot(2,3,2)
ax2.imshow(m_spectrum, cmap='gray')
ax2.title.set_text('FFT of input image')

ax3 = fig.add_subplot(2,3,3)
ax3.imshow(np.log(1+cv2.magnitude(fshift[:,:,0], fshift[:,:,1])), cmap='gray')
ax3.title.set_text('FFT with mask')

ax4 = fig.add_subplot(2,3,4)
ax4.imshow(img_back, cmap='gray')
ax4.title.set_text('After inverse FFT')

ax5 = fig.add_subplot(2,3,5)
ax5.imshow(img_back2, cmap='gray')
ax5.title.set_text('Inverse FFT with inverted mask')

ax6 = fig.add_subplot(2,3,6)
ax6.imshow(img_back3, cmap='gray')
ax6.title.set_text('Band pass filter')

plt.tight_layout()
plt.show()




