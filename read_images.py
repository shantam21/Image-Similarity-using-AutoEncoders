import glob
from pathlib import Path
import cv2 as cv2
import numpy as np

def read_images(directory, test = False):
    """Reading image into numpy arrays

    Args:
        directory (str): Path to the directory
        test (bool, optional): Flag to select test directory. Defaults to False.
    """    
    dir = Path(directory)
    imgs= dir.glob('*jpg')
    data = []
    images = []
    for img in imgs:
        images.append(str(img))
        img = cv2.imread(str(img))
        img = cv2.resize(img, (224,224))
        if img.shape[2] ==1:
            img = np.dstack([img, img, img])
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)/255.
        data.append(img)
        if len(data)%500==0:
            print("batch complete")
    X = np.array(data)
    print("Total number of  examples: ", X.shape)
    if test == False:
        np.save('X.npy', X)
        print("The features are saved in X.npy ...")
    else:
        np.save('X_test.npy',X)
        print("The test features are saved in X_test.npy ...")
    print("Image read process complete ...")
