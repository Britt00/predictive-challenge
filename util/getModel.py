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
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG19
import numpy as np
import matplotlib.pyplot as plt


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
    outputs = Dense(205, activation="relu")(outputs)                      # <-- maybe add this layer later, after proof of concept
    #outputs = Dense(205, activation="relu")(outputs)
    outputs = Dense(nr_of_classes, activation="softmax")(outputs)

    model = Model(inputs, outputs, name="Classifier")
    return model


def visualize_model(model, img, classes, nr_layers=22):
    # transforming the image to an acceptabel input for the model
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    
    # create a model that only has its convolutional layer activations as outputs
    layer_outputs = [layer.output for layer in model.layers[:nr_layers]]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)
    
    # extract the layer names of the model
    layer_names = [layer.name for layer in model.layers[:nr_layers]]    
    
    # plot the actual picture
    plt.imshow(img, aspect="equal")
    plt.title("Original picture")
    
    
    # Plotting the prediction of the model
    pr = model.predict(img_tensor)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(classes, pr[0])
    plt.title("Prediction of the model")
    print(f"This image is of the class '{classes[np.argmax(pr)]}'.")
    
    inputactivation = activations[0]
    size = img.shape[0]
    display_grid = np.zeros((size, size*3))
    for i in range(3):
        channel_image = inputactivation[0, :, :, i]
        display_grid[:, i*size : (i+1)*size] = channel_image
    plt.figure(figsize=(display_grid.shape[1]/23, display_grid.shape[0]/23))
    plt.title(layer_names[0])
    plt.imshow(display_grid, aspect='auto', cmap='inferno')
    
    # plot the activations in a grid
    for layer_name, layer_activation in zip(layer_names[1:], activations[1:]):
        # This is the number of features in the feature map
        n_features = layer_activation.shape[-1]

        # The feature map has shape (1, size, size, n_features)
        size = layer_activation.shape[1]

        # We will tile the activation channels in this matrix
        images_per_row = 16
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size*n_cols, images_per_row*size))
        
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col*images_per_row + row]

                # Post-process the feature to make it somewhat visually apealing
                channel_image -= channel_image.mean()
                channel_image /= (channel_image.std() + 0.0001)
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col+1) * size,
                             row * size : (row+1) * size] = channel_image

        # display the grid
        scale = 1. / size
        plt.figure(figsize=(scale*display_grid.shape[1],
                            scale*display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='inferno')

    plt.show()
    return 1