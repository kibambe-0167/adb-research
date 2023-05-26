from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
from keras.applications import resnet50
import tensorflow as tf
from PIL import Image, ImageEnhance
# 
from skimage.color import rgb2gray
import pandas as pd
import numpy as np
import os
# 
img_data_gen_obj = ImageDataGenerator()

#
# 

# class image for data data standardization
# - data augmentation.
# - data standardization.
class DataStandardization():
    def __init__(self, image) -> None:
        self.image = image
        self.image_size_x = 250
        self.image_size_y = 250
        self. root_path = "./data/faces/img_align_celeba/img_align_celeba/"
        self.data_gen = ImageDataGenerator(
            # 
            rotation_range=10,  # 
            width_shift_range=0.1,  
            height_shift_range=0.1, 
            shear_range=0.2,  
            zoom_range=0.2,  
            horizontal_flip=True, 
            fill_mode='nearest'
        )
    
    def load_image_data_gen(self, ):
        train_generator = self.data_gen.flow_from_directory(
            root_path,
            target_size=(self.image_size_y, self.image_size_x),
            batch_size=32,
            # os.path.join(root_path),
            # class_mode='binary'
        )
        return train_generator
    
    def load_image(self, image_id):
        '''load an image from file. receive the image path.'''
        path = os.path.join(root_path, image_id)
        return load_img(path)
    
    def resize_image(self, target_size):
        '''receives a tuple of image size e.g (200, 200)'''
        # self.image = self.image.resize(target_size)
        return self.image.resize(target_size)
        
    def image_to_array(self,):
        '''convert an image to array.'''
        return img_to_array(self.image)
    
    def resize_image_in_batch(self, images):
        '''resize a lot of images in batches'''
        t_size = (200, 200)
        b_size = 3
        result_images = []
        for i in range(0, len(images), b_size):
            batch = images[i:i+b_size]
            resized_batch = [img.resize(t_size) for img in batch]
            result_images.extend(resized_batch)
        return result_images
    
    def reshape_image(self, image, num_samples):
        '''receives an image and number sample in DS. reshape the image'''
        return image.reshape((num_samples,)+image.shape)

    def filter_corrupted_image(self, root_path, image_path):
        '''filter corrupted images. return true or false''' 
        res = None
        try:
            with open(os.path.join(root_path,image_path), 'rb') as obj:
                res =  tf.compat.as_bytes("JFIF") in obj.peek(20)
                return res
        except IOError as ioe:
            print(ioe)
            return res

    def normalised_pixel_values(self, image, value=250.0 ):
        '''normalise pixels values'''
        return image.astype(np.float32) / value
    
    def convert_image_grey_scale(self, image):
        '''receive an image and convert it to grey scale.'''
        return rgb2gray(image)
    
    def convert_image_grey_scale_array(self, image_array):
        '''receive an image array.
        normalise it values, before converting to grey scale.'''
        return rgb2gray( self.normalised_pixel_values(image_array))
    
    def enhance_image_contrast(self, image, factor = 1.5):
        '''enhance an images contrast. image must not be an array.'''
        # value of > 1, means increase contrast. 
        # value < 1 decrease contrast. value 1, dont change contrast.
        return ImageEnhance.Contrast(image).enhance(factor)
    
    def norm_mean_substraction(self, image):
        '''apply means substraction on the image, image must be in shape of, bath, h, w, channels'''
        return preprocess_input(image)
    
    def save_image(sefl, image:Image):
        '''save an image.'''
        # image.save()
        
    def norm_z_score(self, image_array):
        '''perform z score on the image array'''
        return image_array / np.std(image_array) 
        
    def class_pipeline(self,):
        '''pipe methods for image preprocessing.'''
        pass