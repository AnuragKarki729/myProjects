import numpy as np
from keras.applications import VGG16, InceptionV3
from keras.models import Model
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras.applications import ResNet50
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from keras.applications.inception_v3 import preprocess_input as preprocess_input_incep
from os import listdir
from os.path import join
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import log_loss, accuracy_score

# Define directories for the dataset
train_dir = './TrainTest/TrainSubset'
valid_dir = './TrainTest/ValidationSubset'

# Set constants
INPUT_SIZE_VGG = 224
INPUT_SIZE_INCEPTION = 299
SEED = 1987

# Extract labels from folder names
def extract_labels_from_folders(data_dir):
    folder_names = [d for d in listdir(data_dir) if not d.startswith('.')]
    labels = {folder: idx for idx, folder in enumerate(folder_names)}
    return labels, folder_names

labels_dict, breed_list = extract_labels_from_folders(train_dir)
print(f"Number of classes: {len(labels_dict)}")

# Update labels for all classes
def get_image_labels(data_dir, labels_dict):
    images = []
    labels = []
    for breed, idx in labels_dict.items():
        breed_dir = join(data_dir, breed)
        for img_name in listdir(breed_dir):
            if img_name.lower().endswith('.jpg'):
                images.append(img_name)
                labels.append(idx)
    return images, labels

train_images, train_labels = get_image_labels(train_dir, labels_dict)
valid_images, valid_labels = get_image_labels(valid_dir, labels_dict)
train_labels = np.array(train_labels)
valid_labels = np.array(valid_labels)

# Prepare label binarizer
lb = LabelBinarizer()
lb.fit(train_labels)
y_train = lb.transform(train_labels)
y_valid = lb.transform(valid_labels)

# Helper function to read and preprocess images
def read_img(img_id, size, preprocess_func, data_dir, labels_dict):
    breed_idx = None
    for breed, idx in labels_dict.items():
        breed_dir = join(data_dir, breed)
        if img_id in listdir(breed_dir):
            breed_idx = idx
            break
    if breed_idx is None:
        raise FileNotFoundError(f"Image {img_id} not found in {data_dir}")

    img_path = join(data_dir, breed_list[breed_idx], img_id)
    img = image.load_img(img_path, target_size=size)
    img = image.img_to_array(img)
    img = preprocess_func(np.expand_dims(img.copy(), axis=0))
    return img


def extract_features(model, img_list, input_size, preprocess_func, data_dir):
    features = np.zeros((len(img_list), input_size, input_size, 3), dtype='float32')
    for i, img_id in tqdm(enumerate(img_list), total=len(img_list)):
        img = read_img(img_id, (input_size, input_size), preprocess_func, data_dir, labels_dict)
        features[i] = img
    return features

x_train_vgg = extract_features(VGG16, train_images, INPUT_SIZE_VGG, preprocess_input_vgg, train_dir)
x_valid_vgg = extract_features(VGG16, valid_images, INPUT_SIZE_VGG, preprocess_input_vgg, valid_dir)


# x_train_resnet = extract_features(ResNet50, train_images, INPUT_SIZE_INCEPTION, preprocess_input_resnet, train_dir)
# x_valid_resnet = extract_features(ResNet50, valid_images, INPUT_SIZE_INCEPTION, preprocess_input_resnet, valid_dir)

# x_train_incep = extract_features(InceptionV3, train_images, INPUT_SIZE_INCEPTION, preprocess_input_incep, train_dir)
# x_valid_incep = extract_features(InceptionV3, valid_images, INPUT_SIZE_INCEPTION, preprocess_input_incep, valid_dir)

# # Define and compile VGG16 model
def create_model_vgg16(input_shape, num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=input_shape)
    x = base_model.output
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model_vgg16 = create_model_vgg16((INPUT_SIZE_VGG, INPUT_SIZE_VGG, 3), len(labels_dict))

# Define ModelCheckpoint callback for VGG16
checkpoint_vgg16 = ModelCheckpoint(
    'DogModelVGG16.keras',
    monitor='accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Train the VGG16 model
history_vgg16 = model_vgg16.fit(
    x_train_vgg, y_train,
    validation_data=(x_valid_vgg, y_valid),
    epochs=5,
    batch_size=64,
    callbacks=[checkpoint_vgg16]
)

# Evaluate VGG16 model
valid_probs_vgg16 = model_vgg16.predict(x_valid_vgg)
valid_preds_vgg16 = np.argmax(valid_probs_vgg16, axis=1)
print(f'Validation VGG LogLoss: {log_loss(y_valid, valid_probs_vgg16)}')
print(f'Validation VGG Accuracy: {accuracy_score(np.argmax(y_valid, axis=1), valid_preds_vgg16)}')

# Save VGG16 model
model_vgg16.save('DogModelVGG16.keras')

# # Define and compile InceptionV3 model
# def create_model_inception(input_shape, num_classes):
#     base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg', input_shape=input_shape)
#     x = base_model.output
#     x = Dense(1024, activation='relu')(x)
#     predictions = Dense(num_classes, activation='softmax')(x)
#     model = Model(inputs=base_model.input, outputs=predictions)
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# model_inception = create_model_inception((INPUT_SIZE_INCEPTION, INPUT_SIZE_INCEPTION, 3), len(labels_dict))

# # Define ModelCheckpoint callback for InceptionV3
# checkpoint_inception = ModelCheckpoint(
#     'DogModelInception.keras',
#     monitor='accuracy',
#     save_best_only=True,
#     mode='max',
#     verbose=1
# )

# # Train the InceptionV3 model
# history_inception = model_inception.fit(
#     x_train_incep, y_train,
#     validation_data=(x_valid_incep, y_valid),
#     epochs=10,
#     batch_size=64,
#     callbacks=[checkpoint_inception]
# )

# # Evaluate InceptionV3 model
# valid_probs_inception = model_inception.predict(x_valid_incep)
# valid_preds_inception = np.argmax(valid_probs_inception, axis=1)
# print(f'Validation Inception LogLoss: {log_loss(y_valid, valid_probs_inception)}')
# print(f'Validation Inception Accuracy: {accuracy_score(np.argmax(y_valid, axis=1), valid_preds_inception)}')

# # Save InceptionV3 model
# model_inception.save('DogModelInception.keras')

# def create_model_resnet50(input_shape, num_classes):
#     base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=input_shape)
#     x = base_model.output
#     x = Dense(1024, activation='relu')(x)
#     predictions = Dense(num_classes, activation='softmax')(x)
#     model = Model(inputs=base_model.input, outputs=predictions)
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# checkpoint_resnet = ModelCheckpoint(
#     'DogModelResNet50.keras',
#     monitor='accuracy',
#     save_best_only=True,
#     mode='max',
#     verbose=1
# )

# # Define and compile ResNet50 model
# model_resnet50 = create_model_resnet50((INPUT_SIZE_INCEPTION, INPUT_SIZE_INCEPTION, 3), len(labels_dict))

# # Train the ResNet50 model
# history_resnet50 = model_resnet50.fit(
#     x_train_resnet, y_train,
#     validation_data=(x_valid_resnet, y_valid),
#     epochs=10,
#     batch_size=64,
#     callbacks=[checkpoint_resnet]
# )

# # Evaluate ResNet50 model
# valid_probs_resnet = model_resnet50.predict(x_valid_resnet)
# valid_preds_resnet = np.argmax(valid_probs_resnet, axis=1)
# print(f'Validation ResNet LogLoss: {log_loss(y_valid, valid_probs_resnet)}')
# print(f'Validation ResNet Accuracy: {accuracy_score(np.argmax(y_valid, axis=1), valid_preds_resnet)}')

# # Save ResNet50 model
# model_resnet50.save('DogModelResNet50.keras')