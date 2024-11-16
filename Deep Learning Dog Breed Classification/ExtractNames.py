import os

def get_breed_name(train_dir = './TrainTest/TrainSubset'):
    breed_dirs = [f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))]
    class_names = [breed.split('-')[1] for breed in breed_dirs]
    return class_names

