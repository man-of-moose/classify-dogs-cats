# classify-dogs-cats

Flask Application that predicts "Cat vs Dog" given an image.

## Data

Data sourced from Kaggle.com: https://www.kaggle.com/competitions/dogs-vs-cats/data

- 25,000 images, split evenly between cats and dogs
- Image dimensions not standardized


## Steps

### Processing Data

- zipfile package was used to unzip data
- opencv package was used to load image data and convert to image arrays
  - images loaded using Grayscale
  - images normalized between [0,1]
  - images reshaped using np.reshape(-1,100,100,1)
- pickle package was used to save the image data for later use by models

### Model Building

- Following tensorflow classes were used to construct a 5-layer sequential model
  - Sequential
  - Dense
  - Dropout
  - Conv2D
  - MaxPooling2D
  - Flatten
  - Activation

- Model compilation:
  - optimizer: "adam"
  - loss: "binary_crossentropy"
  - metrics: ["accuracy"]


### Flask App

Basic flask application was built to allow users to enter image urls for prediction. Model.pickle is loaded in app.py and used to predict new downloaded iamges. /predict/ page dynamically loads the prediction confidence, class, and entered image.

![Image](pics/flask.png?raw=true)
