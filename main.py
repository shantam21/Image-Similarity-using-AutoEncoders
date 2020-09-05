import numpy as np
from read_images import read_images
from model import autoencoder_model
from model import train_model
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import CosineSimilarity
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import glob

def plot_top_n_matches(no_of_matches,image_similarities):
    """
    This function plots the top-n images for the test image.

    Args:
        no_of_matches (int): no of matches required
        image_similarities (dict): Image and the similarity w.r.t test image.
    """    
    top_n_matches = sorted(image_similarities.items(), key=lambda x: x[1])[:no_of_matches]
    image_names = [el[0].replace('\\','/') for el in top_n_matches]
    dir = Path(test_directory)
    test_image = dir.glob('*jpg')
    test_image = [el for el in test_image][0]
    rows = round(no_of_matches/5)
    columns = 5
    fig, ax = plt.subplots(rows, columns)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(np.asarray(Image.open(test_image)))
    i = 0
    for x in range(rows):
        for y in range(columns):
            ax[x, y].set_xticks([])
            ax[x, y].set_yticks([])
            if i >= len(image_names):
                ax[x, y].axis('off')
            else:
                ax[x, y].imshow(np.asarray(Image.open(image_names[i])))
            i += 1
    plt.show()
               
               
if __name__ == '__main__':
    
    #  configurations for the script
    no_of_matches = 10
    
    # Image directories
    train_directory = 'data/train'
    test_directory = 'data/test'
    from_pretrained = True
    
    # Read images into np arrays
    if from_pretrained == True:
        read_images(test_directory, test = True)
        X = np.load("X.npy")
        model = load_model('trained_model.h5')
    else:
        read_images(train_directory)
        read_images(test_directory, test = True)
        X = np.load("X.npy")
        train_model(X, epochs = 30, bs = 128, validation_split = 0.2)
        model = load_model('trained_model.h5')
    
    
    encoded_data = model.predict(X)
    np.save('image_encodings.npy', encoded_data)
    
    encodings_train = np.load('image_encodings.npy')
    encodings_test = np.load('X_test.npy')
    dir = Path(train_directory)
    imgs=dir.glob('*jpg')
    names = [str(el) for el in imgs]
    
    #  Cosine similarities of test image w.r.t train images
    similarities = []
    cosine_sim = CosineSimilarity(axis = 1)
    for i in range(encodings_train.shape[0]):
        sim = cosine_sim(encodings_test[0], encodings_train[i]).numpy()
        # print(sim)
        similarities.append(sim)
    
    image_similarities = dict(zip(names, similarities))
    plot_top_n_matches(no_of_matches,image_similarities)

    
    