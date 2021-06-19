import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

train_cnn = pd.read_csv("data/training.csv")
test_cnn = pd.read_csv("data/test.csv")


def string2image(string):
    """Converts a string to a numpy array."""
    return np.array([int(item) for item in string.split()]).reshape((96, 96))

def plot_faces(nrows=5, ncols=5):
    """Randomly displays some faces from the training data."""
    selection = np.random.choice(train_cnn.index, size=(nrows*ncols), replace=False)
    image_strings = train_cnn.loc[selection]['Image']
    fig, axes = plt.subplots(figsize=(10, 10), nrows=nrows, ncols=ncols)
    for string, ax in zip(image_strings, axes.ravel()):
        ax.imshow(string2image(string), cmap='gray')
        ax.axis('off')
    plt.show()

plot_faces()

def facial_keypoints(index):
    keypoint_cols = list(train_cnn.columns)[:-1]

    xy = train_cnn.iloc[0][keypoint_cols].values.reshape((15, 2))

    plt.plot(xy[:, 0], xy[:, 1], 'ro')
    plt.imshow(string2image(train_cnn.iloc[index]['Image']), cmap='gray')
    plt.show()

facial_keypoints(0)

def plot_faces_with_keypoints(nrows=5, ncols=5):
    """Randomly displays some faces from the training data with their keypoints."""
    selection = np.random.choice(train_cnn.index, size=(nrows*ncols), replace=False)
    image_strings = train_cnn.loc[selection]['Image']
    keypoint_cols = list(train_cnn.columns)[:-1]
    keypoints = train_cnn.loc[selection][keypoint_cols]
    fig, axes = plt.subplots(figsize=(10, 10), nrows=nrows, ncols=ncols)
    for string, (iloc, keypoint), ax in zip(image_strings, keypoints.iterrows(), axes.ravel()):
        xy = keypoint.values.reshape((15, 2))
        ax.imshow(string2image(string), cmap='gray')
        ax.plot(xy[:, 0], xy[:, 1], 'ro')
        ax.axis('off')
    plt.show()

plot_faces_with_keypoints()

def preprocess_data():
    fully_annotated = train_cnn.dropna()
    X = np.stack([string2image(string) for string in fully_annotated['Image']]).astype(np.float)[:, :, :, np.newaxis]
    y = np.vstack(fully_annotated[fully_annotated.columns[:-1]].values)

    print("X.shape, X.dtype", X.shape, X.dtype)
    print("y.shape, y.dtype", y.shape, y.dtype)

    X_train = X / 255.

    output_pipe = make_pipeline(MinMaxScaler(feature_range=(-1, 1)))
    y_train = output_pipe.fit_transform(y)

preprocess_data()
