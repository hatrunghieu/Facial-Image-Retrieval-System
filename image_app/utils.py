from . import spark_utils
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from django.conf import settings
from tensorflow.keras.preprocessing import image


filter_path = os.path.join(settings.BASE_DIR, 'image_app',
                           'data', 'haarcascades', 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(filter_path)
target_size = (200, 200)


def process_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes = face_cascade.detectMultiScale(image_rgb, 1.3, 5)

    for i, box in enumerate(bboxes):
        x, y, w, h = box
        face = image[y:y+h, x:x+w]
        face_resized = cv2.resize(face, target_size)

    return face_resized


model = load_model(os.path.join(settings.BASE_DIR,
                                'image_app/data/ml_models/embedding_model.keras'))


def extract_embeddings(img):

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Generate the embedding
    features = model.predict(x)
    features_flat = features.flatten()

    return features_flat.tolist()


def query_database(query_embedding):
    neighbors = spark_utils.perform_query(query_embedding)
    results = [neighbor.asDict() for neighbor in neighbors]
    for result in results:
        result['name'] = result['image_path'].split('/')[3].replace('_', ' ')
        result['distance'] = round(result['distCol'], 5)
    return results
