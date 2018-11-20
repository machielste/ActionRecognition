import numpy as np
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.preprocessing import image
import time
from PIL import Image
import io



class Extractor():
    ##Convert a single video frame to a vector, this vectors is the output of the second last layer of inceptionV3
    def __init__(self, ):
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


        start_time = time.time()
        features = self.model.predict(x)
        print("--- %s seconds ---" % (time.time() - start_time))

        features = features[0]
        return features


x = Extractor()

image = Image.open("tests/cat.jpeg")
features = (x.extract(image))
print(len(features))