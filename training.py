import os
import numpy as np
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing import image

embedder = FaceNet()
dataset_path = "dataset"
embeddings = []
names = []

for img_name in os.listdir(dataset_path):
    path = os.path.join(dataset_path, img_name)
    img = image.load_img(path, target_size=(160, 160))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    embed = embedder.embeddings(img_array)[0]
    embeddings.append(embed)

    label = img_name.split("_")[0]
    names.append(label)

# Save embeddings and labels
np.save("embeddings.npy", np.array(embeddings))
np.save("labels.npy", LabelEncoder().fit_transform(names))
print("[✓] Embeddings and labels saved.")
