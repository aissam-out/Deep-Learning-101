import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model

# the test image
image = cv2.imread('images/obama.png')

# Convert the image to RGB colorspace
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plot our image
fig = plt.figure(figsize = (9,9))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('image')
ax1.imshow(image)
#plt.show()

# Use the face detection code with our trained conv-net
def plot_keypoints(img_path, face_cascade_path, model_path, scale=1.2, neighbors=5, key_size=10):

    face_cascade=cv2.CascadeClassifier(face_cascade_path)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scale, neighbors)
    fig = plt.figure(figsize=(40, 40))
    ax = fig.add_subplot(121, xticks=[], yticks=[])
    ax.set_title('Image with Facial Keypoints')

    # Print the number of faces detected in the image
    print('Number of faces detected:', len(faces))

    # Make a copy of the orginal image to draw face detections on
    image_with_detections = np.copy(img)

    # Get the bounding box for each detected face
    for (x,y,w,h) in faces:
        # Add a red bounding box to the detections image
        cv2.rectangle(image_with_detections, (x,y), (x+w,y+h), (255,0,0), 3)
        bgr_crop = image_with_detections[y:y+h, x:x+w]
        orig_shape_crop = bgr_crop.shape
        gray_crop = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
        resize_gray_crop = cv2.resize(gray_crop, (96, 96)) / 255
        model = load_model(model_path)
        landmarks = np.squeeze(model.predict(
            np.expand_dims(np.expand_dims(resize_gray_crop, axis=-1), axis=0)))
        ax.scatter(((landmarks[0::2] * 48 + 48)*orig_shape_crop[0]/96)+x,
                   ((landmarks[1::2] * 48 + 48)*orig_shape_crop[1]/96)+y,
                   marker='o', c='c', s=key_size)

    ax.imshow(cv2.cvtColor(image_with_detections, cv2.COLOR_BGR2RGB))
    plt.show()

# Paint the predicted keypoints on the test image
obamas = plot_keypoints('images/test3.jpg',
                        'haar_detectors/haarcascade_frontalface_default.xml',
                        'my_model_final.h5')
