import tensorflow as tf
import cv2
import numpy as np
from typing import List
import glob
import os

class DIGITS_CLASSIFIER():
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def preprocess_images(self,images:List[np.ndarray]):
        gray_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
        resized_images = np.array([cv2.resize(gray_image, (32, 32)) for gray_image in gray_images])
        normalized_images = resized_images.astype('float32') / 255.0
        return np.expand_dims(normalized_images, axis=-1)
    
    def postprocess_predictions(self, predictions):
        predicted_labels = [''.join(map(str, row)) for row in np.argmax(predictions, axis=2)]
        predicted_confs = np.min(np.max(predictions, axis=2),axis=1)
        return predicted_labels, predicted_confs
    
    def predict(self, images:List[np.ndarray]):
        preprocess_images = self.preprocess_images(images)
        predictions = self.model.predict(preprocess_images)
        return self.postprocess_predictions(predictions)
    
    def predict_without_preprocess(self, preprocessed_images):
        predictions = self.model.predict(preprocessed_images)
        return self.postprocess_predictions(predictions)
    
    def _read_images_from_directory(self, directory):
        pattern = os.path.join(directory, "*.*")
        png_files = glob.glob(pattern)
        return [cv2.imread(file) for file in png_files]
    
    def predict_from_directory(self, directory):
        images = self._read_images_from_directory(directory)
        return self.predict(images)
    


# example
if __name__ == '__main__':
    digits_classifier = DIGITS_CLASSIFIER(model_path='models/svhn_2digits_model.keras')

    random_array = np.random.rand(1000,32,32,1).astype(np.float32)
    predicted_labels, predicted_confs = digits_classifier.predict_without_preprocess(random_array)
    print(predicted_labels, predicted_confs)

    predicted_labels, predicted_confs = digits_classifier.predict_from_directory('custom_images')
    print(predicted_labels, predicted_confs)
