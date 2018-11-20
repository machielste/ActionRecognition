import numpy as np
from keras.applications.vgg19 import VGG19, preprocess_input


class Extractor():
    ##Convert a single video frame to a vector, this vectors is the output of the second last layer of inceptionV3
    def __init__(self):
        self.model = VGG19(
            weights='imagenet',
            include_top=False,
            pooling="avg"
        )

    def extract(self, image_path):
        x = image_path
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Get the prediction.

        features = self.model.predict(x)
        features = features[0]
        return features
