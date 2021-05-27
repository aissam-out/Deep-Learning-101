import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('images/test3.jpg')

def edge_detector(image, display=False):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    orig_img = np.copy(gray)
    kernel = np.ones((4,4),np.float32)/16
    blur = cv2.filter2D(orig_img,-1,kernel)

    # Perform Canny edge detection on blurred image
    edges_blur = cv2.Canny(blur,100,200)

    # Dilate the image to amplify edges
    edges_blur = cv2.dilate(edges_blur, None)

    if (display == True):
        # Display the image with the detections
        plt.figure()
        plt.title('Blur')
        plt.imshow(blur)
        plt.show()

        plt.figure()
        plt.title('edge Blur')
        plt.imshow(edges_blur)
        plt.show()

    return edges_blur, blur

edge_detector(image, True)
