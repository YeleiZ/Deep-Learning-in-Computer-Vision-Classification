from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
import os.path as osp
import json
from PIL import Image
import tensorflow
from utils import Config
import tensorflow as tf


class polyvore_dataset:
    def __init__(self):
        self.root_dir = Config['root_path']
        self.image_dir = osp.join(self.root_dir, 'images')
        self.transforms = self.get_data_transforms()

    #crop and nomalize image
    def get_data_transforms(self):
        data_transforms = {
            'train': transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
        }
        return data_transforms

    def create_dataset(self):
        x = []
        y = []
        f = open("/home/ubuntu/polyvore_outfits/pairwise_compatibility_train.txt", 'r')
        for row in f:
            imageID = row.split()[1] + '.jpg' + '   ' + row.split()[2] + '.jpg'
            x.append(imageID)
            y.append(int(row.split()[0]))
        f.close()
        print('len of X: {}, # of categories: {}'.format(len(x), 1))

        # Use 10% of the data and split dataset
        num = len(y) // 10
        X_train, X_test, y_train, y_test = train_test_split(x[:num],  y[:num], test_size=0.2)
        return X_train, X_test, y_train, y_test, 1



class DataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, dataset, dataset_size, params):
        self.batch_size = params['batch_size']
        self.shuffle = params['shuffle']
        self.n_classes = params['n_classes']
        self.X, self.y, self.transform = dataset
        self.image_dir = osp.join(Config['root_path'], 'images')
        self.on_epoch_end()
        self.X1 = []
        self.X2 = []
        for row in self.X:
            image = row.split()
            self.X1.append(image[0])
            self.X2.append(image[1])

    def __len__(self):
        return int(np.floor(len(self.X)/self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index+1) * self.batch_size]
        X1, X2, y = self.__data_generation(indexes)
        X1, X2, y = np.stack(X1), np.stack(X2), np.stack(y)
        X1 = np.moveaxis(X1, 1, 3)
        X2 = np.moveaxis(X2, 1, 3)
        X = tf.concat([X1, X2], axis=-1)
        #print(X.shape,y.shape)
        return X, y

    # Generates data containing batch_size samples
    def __data_generation(self, indexes):
        X1 = []; X2 = []; y = []
        for idx in indexes:
            file_path1 = osp.join(self.image_dir, self.X1[idx])
            file_path2 = osp.join(self.image_dir, self.X2[idx])
            X1.append(self.transform(Image.open(file_path1)))
            X2.append(self.transform(Image.open(file_path2)))
            y.append(self.y[idx])
        return X1, X2, y

    # Updates indexes after each epoch
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)




class PredictDataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, dataset, dataset_size, params):
        self.batch_size = params['batch_size']
        self.shuffle = params['shuffle']
        self.n_classes = params['n_classes']
        self.X, self.y, self.transform = dataset
        self.image_dir = osp.join(Config['root_path'], 'images')
        self.on_epoch_end()
        self.X1 = []
        self.X2 = []
        for row in self.X:
            image = row.split()
            self.X1.append(image[0])
            self.X2.append(image[1])

    def __len__(self):
        return int(np.floor(len(self.X)/self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index+1) * self.batch_size]
        X1, X2 = self.__data_generation(indexes)
        X1, X2 = np.stack(X1), np.stack(X2)
        X1 = np.moveaxis(X1, 1, 3)
        X2 = np.moveaxis(X2, 1, 3)
        X = tf.concat([X1, X2], axis=-1)
        return X

    # Generates data containing batch_size samples
    def __data_generation(self, indexes):
        X1 = []; X2 = []
        for idx in indexes:
            file_path1 = osp.join(self.image_dir, self.X1[idx])
            file_path2 = osp.join(self.image_dir, self.X2[idx])
            X1.append(self.transform(Image.open(file_path1)))
            X2.append(self.transform(Image.open(file_path2)))
        return X1, X2

    # Updates indexes after each epoch
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
