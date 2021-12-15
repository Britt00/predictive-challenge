"""
PreProcessing.py

Contains all functions for loading and preprocessing data
for use in Tensorflow Keras purposes.

Needs some cleaning up
David 15/12/2021
"""

from pycocotools.coco import COCO
import numpy as np
from itertools import repeat
import random
#import os
import cv2

#for augmentations
from tensorflow.keras.preprocessing.image import ImageDataGenerator

## For visualizing results
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def zippedshuffle(imgs, labels):
    c = list(zip(imgs, labels))
    random.shuffle(c)
    imgs, labels = zip(*c)
    return list(imgs), list(labels)

def filterDataset(folder, classes=None, mode='train'):    
    # initialize COCO api for instance annotations
    annFile = '{}/annotations/instances_{}.json'.format(folder, mode)
    coco = COCO(annFile)
    
    images = []
    labels = []
    if classes!=None:
        # iterate for each individual class in the list
        for idc, className in enumerate(classes):
            # get all images containing given categories
            catIds = coco.getCatIds(catNms=className)
            imgIds = coco.getImgIds(catIds=catIds)
            images += coco.loadImgs(imgIds)
            
            ### one hot encoding labeling ###
            classlabel = np.zeros(len(classes)).tolist()
            classlabel[idc] = 1
            labels.extend(repeat(classlabel, len(imgIds)))
    
    else:
        imgIds = coco.getImgIds()
        images = coco.loadImgs(imgIds)
        labels = [0]*len(images)
    
    # Now, filter out the repeated images
    unique_images = []
    true_labels = []
    for i in range(len(images)):
        if images[i] not in unique_images:
            unique_images.append(images[i])
            true_labels.append(labels[i])
    
    ### shuffle ###
    unique_images, true_labels = zippedshuffle(unique_images, true_labels)
    
    dataset_size = len(unique_images)
    
    return list(unique_images), list(true_labels), dataset_size, coco

def dataGeneratorCoco(images, labels, coco, folder, 
                      input_image_size=(224,224), batch_size=4, mode='train'):
    
    img_folder = '{}/images/{}'.format(folder, mode)
    dataset_size = len(images)
    catIds = coco.getCatIds(catNms=classes)
    
    c = 0
    while(True):
        img = np.zeros((batch_size, input_image_size[0], input_image_size[1], 3)).astype('float')
        lbl = np.zeros((batch_size, len(classes)))
#         mask = np.zeros((batch_size, input_image_size[0], input_image_size[1], 1)).astype('float')

        for i in range(c, c+batch_size): #initially from 0 to batch_size, when c = 0
            imageObj = images[i]
            
            ### Retrieve Image ###
            train_img = getImage(imageObj, img_folder, input_image_size)
            
#             ### Create Mask ###
#             if mask_type=="binary":
#                 train_mask = getBinaryMask(imageObj, coco, catIds, input_image_size)
            
#             elif mask_type=="normal":
#                 train_mask = getNormalMask(imageObj, classes, coco, catIds, input_image_size)     

            ### Retrieve Class label ###
            train_label = labels[i]
    
            # Add to respective batch sized arrays
            img[i-c] = train_img
            lbl[i-c] = train_label
            
        c+=batch_size
        if(c + batch_size >= dataset_size):
            c=0
            images, labels = zippedshuffle(images, labels)
            
        lbl.flatten()
        yield img, lbl #, mask
        
 # def getNormalMask(imageObj, classes, coco, catIds, input_image_size):
#     annIds = coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
#     anns = coco.loadAnns(annIds)
#     cats = coco.loadCats(catIds)
#     train_mask = np.zeros(input_image_size)
#     for a in range(len(anns)):
#         className = getClassName(anns[a]['category_id'], cats)
#         pixel_value = classes.index(className)+1
#         new_mask = cv2.resize(coco.annToMask(anns[a])*pixel_value, input_image_size)
#         train_mask = np.maximum(new_mask, train_mask)

#     # Add extra dimension for parity with train_img size [X * X * 3]
#     train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
#     return train_mask  
    
# def getBinaryMask(imageObj, coco, catIds, input_image_size):
#     annIds = coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
#     anns = coco.loadAnns(annIds)
#     train_mask = np.zeros(input_image_size)
#     for a in range(len(anns)):
#         new_mask = cv2.resize(coco.annToMask(anns[a]), input_image_size)
        
#         #Threshold because resizing may cause extraneous values
#         new_mask[new_mask >= 0.5] = 1
#         new_mask[new_mask < 0.5] = 0

#         train_mask = np.maximum(new_mask, train_mask)

#     # Add extra dimension for parity with train_img size [X * X * 3]
#     train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
#     return train_mask
        
def visualizeGenerator(gen):
    img, mask = next(gen)
    
    fig = plt.figure(figsize=(20, 10))
    outerGrid = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.1)
    
    for i in range(2):
        innerGrid = gridspec.GridSpecFromSubplotSpec(2, 2,
                        subplot_spec=outerGrid[i], wspace=0.05, hspace=0.05)

        for j in range(4):
            ax = plt.Subplot(fig, innerGrid[j])
            if(i==1):
                ax.imshow(img[j])
            else:
                ax.imshow(mask[j][:,:,0])
                
            ax.axis('off')
            fig.add_subplot(ax)        
    plt.show()
    
def augmentationsGenerator(gen, augGeneratorArgs, seed=None):
    # Initialize the image data generator with args provided
    image_gen = ImageDataGenerator(**augGeneratorArgs)
    
    # Remove the brightness argument for the mask. Spatial arguments similar to image.
    augGeneratorArgs_mask = augGeneratorArgs.copy()
    _ = augGeneratorArgs_mask.pop('brightness_range', None)
    # Initialize the mask data generator with modified args
    mask_gen = ImageDataGenerator(**augGeneratorArgs_mask)
    
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    
    for img, mask in gen:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation of the images 
        # will end up different from the augmentation of the masks
        g_x = image_gen.flow(255*img, 
                             batch_size = img.shape[0], 
                             seed = seed, 
                             shuffle=True)
        g_y = mask_gen.flow(mask, 
                             batch_size = mask.shape[0], 
                             seed = seed, 
                             shuffle=True)
        
        img_aug = next(g_x)/255.0
        
        mask_aug = next(g_y)
                   

        yield img_aug, mask_aug