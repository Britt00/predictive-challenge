"""
getModel.py

Contains all functions for creating the tf model.
for use in Tensorflow Keras purposes.

Maybe the build function should allow for the specification
of the outputlayers?
David 15/12/2021

"""

from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19

def build_model(input_shape, nr_of_classes):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained VGG19 Model """
    vgg19 = VGG19(include_top=False, weights="imagenet", input_tensor=inputs)
    for layer in vgg19.layers:
        layer.trainable = False
    
    """ Output layers """
    s1 = vgg19.get_layer("block5_pool").output         ## last layer output data    
    outputs = Flatten()(s1)
    #outputs = Dense(4096, activation="relu")(outputs)                      # <-- maybe add this layer later, after proof of concept
    #outputs = Dense(4096, activation="relu")(outputs)
    outputs = Dense(nr_of_classes, activation="softmax")(outputs)

    model = Model(inputs, outputs, name="Classifier")
    return model