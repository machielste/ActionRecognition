import numpy as np
from PIL import Image
from keras.applications.vgg19 import VGG19

class Extractor():
    # Convert a single video frame to a vector, this vectors is the output of the second last layer of VGG19
    def __init__(self):
        self.model = VGG19(
            weights='imagenet',
            include_top=False,
            pooling="avg"
        )

    def extract(self, image_path):
        # Using pillow to reszie image as keras preprocessing is slow
        x = Image.fromarray(image_path)
        x = x.resize([224, 224])
        x = np.expand_dims(x, axis=0)

        features = self.model.predict(x)

        features = features[0]
        return features
