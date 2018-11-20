from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model, save_model
import numpy as np



# Get model with pretrained weights.
base_model = InceptionV3(
    weights='imagenet',
    include_top=True
)

# extract features at the final pool layer.
model = Model(
    inputs=base_model.input,
    outputs=base_model.get_layer('avg_pool').output
)

model.save("model.h5")