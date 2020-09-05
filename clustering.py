from keras import backend as K
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from pathlib import Path
from yellowbrick.cluster import KElbowVisualizer


if __name__ == '__main__':
    train_directory = 'data/train'
    X = np.load('image_encodings.npy')
    model = load_model('model.h5')
    get_encoded = K.function([model.layers[0].input], [model.layers[4].output])
    X_encoded = np.empty((len(X), 56, 56, 16), dtype='float32')
    step = 100
    for i in range(0, len(X), step):
        x_batch = get_encoded([X[i:i+step]])[0]
        X_encoded[i:i+step] = x_batch

    print(X_encoded.shape)
    X_encoded_reshape = X_encoded.reshape(X_encoded.shape[0], X_encoded.shape[1]*X_encoded.shape[2]*X_encoded.shape[3])
    print(X_encoded_reshape.shape)
    df = pd.DataFrame(X_encoded_reshape)
    dir = Path(train_directory)
    images = dir.glob('*jpg')
    images = [el for el in images]
    
    df['image_name'] = images
    # df.to_csv('op_encoded.csv')
    
    pca = PCA(0.95)
    x = df.drop(['image_name'], axis= 1)
    pca.fit(x)
    x = pca.transform(x)
    # kmeans_model = KElbowVisualizer(KMeans(), k=10)
    # kmeans_model.fit(x)
    # kmeans_model.show()
    km = KMeans(n_clusters=5)
    km.fit(x)
    clustering_result = pd.DataFrame(columns = ['Image', 'Label'])
    clustering_result['Image'] = images
    clustering_result['Label'] = km.labels_
    clustering_result.to_csv('clusters.csv', index = None)
     