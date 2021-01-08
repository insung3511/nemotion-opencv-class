import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure

image = plt.imread('cherrybomb.png')

fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16,16), 
        cells_per_block=(1,1), visualize=True, multichannel=True)

flag, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input Image')

hog_img_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_img_rescaled, cmap=plt.cm.gray)
ax2.set_title('Final Result')
plt.show()
