import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.figure(figsize=(16, 16))

# Convert image to grayscale
img_gs = cv2.imread('MRI1.jpeg', cv2.IMREAD_GRAYSCALE)


print("Image Properties")
print("- Number of Pixels: " + str(img_gs.size))
print("- Shape/Dimensions: " + str(img_gs.shape))

cv2.imwrite('gs.jpg', img_gs)

# Apply canny edge detector algorithm on the image to find edges
edges = cv2.Canny(img_gs, 100,700)

# Plot the original image against the edges
plt.subplot(121), plt.imshow(img_gs)
plt.title('Original Gray Scale Image')
plt.subplot(122), plt.imshow(edges)
plt.title('Edge Image')

# Display the two images
plt.show()