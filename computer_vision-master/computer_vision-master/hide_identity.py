import cv2
import numpy as np
import face_detect as fd
import matplotlib.pyplot as plt

image = cv2.imread('images/boubker.jpg')

face_cascade = cv2.CascadeClassifier('haar_detectors/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haar_detectors/haarcascade_eye.xml')

image_with_faces, face_crop, faces = fd.face_eye_detector(image, eye=False, display=False)

def hide_identity(image_with_faces, face_crop, faces, display=True):
    ## Blur the bounding box around each detected face using an averaging filter and display the result
    result_image = np.copy(image_with_faces)
    kernel_2 = np.ones((40,40),np.float32)/1600
    blur_2 = cv2.filter2D(face_crop,-1,kernel_2)
    for (x,y,w,h) in faces:
        result_image[y:y+blur_2.shape[0], x:x+blur_2.shape[1]] = blur_2

    if display == True :
        plt.figure()
        plt.title('blurred image')
        plt.imshow(result_image)
        plt.show()

    return result_image

hide_identity(image_with_faces, face_crop, faces)
