# Image-Similarity-using-AutoEncoders

The solution retreives top-n similar images for a test image.
It also clusters the solution into K clusters. Here, k = 5

## Output
For the test image ![test image 1](https://github.com/shantam21/Image-Similarity-using-AutoEncoders/blob/master/outputs/test1.jpg)
The similar images are:![op image 1](https://github.com/shantam21/Image-Similarity-using-AutoEncoders/blob/master/outputs/op_1.png)

## How to run the script
The primary script is main.py. 
1. Place the test image in 'data/test/' (make sure you place the single image)
2. Change the parameter for no_of_matches in *main.py* according to requirement and run.
3. For clustering, after executing *main.py* execute *clustering.py*
parameter: no_of_matches = 10 (defaulted to 10, can be changed later)

## Methodology:
1. The images are raed into numpy arrays which is fed to the autoencoder model.
2. The image encodings are then used to calculate cosine-similarities w.r.t all the train images.
3. Top n images having the highest similarities are selected. *(score closer to -1 has higher similarity)
4. PCA is then used to reduce the dimensionalityy by keeping *95%* of the variance.
5. Using *lbow Mthod*, it is observed that the optimum number of clusters for these set of images is *5*.
6. These cluster information is stored in *clusters.csv*

## Files 
1. *model.h5*: autoencoder model.
2. *trained_model.h5*: trained autoencoder_model
3. *elbow_curve.png*: Elbow curve for optimum number of clusters.
4. *clusters.csv*: cluster info for all images
5. *read_images.py*: conatins unctions to convert image into array
6. *model.py*: contains model architecture
7. *main.py*: contains initializations and similarity calculations.
