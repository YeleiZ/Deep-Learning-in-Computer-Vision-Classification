from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os
import numpy as np
import os.path as osp
import json
from tqdm import tqdm
from PIL import Image
import tensorflow
from utils import Config

class polyvore_dataset:
    def __init__(self):
        self.root_dir = Config['root_path']
        self.image_dir = osp.join(self.root_dir, 'images')
        self.transforms = self.get_data_transforms()
        # self.X_train, self.X_test, self.y_train, self.y_test, self.classes = self.create_dataset()


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
        # map id to category
        meta_file = open(osp.join(self.root_dir, Config['meta_file']), 'r')
        meta_json = json.load(meta_file)
        id_to_category = {}
        for k, v in tqdm(meta_json.items()):
            id_to_category[k] = v['category_id']

        # create X, y pairs, X:ID, y:catagory
        files = os.listdir(self.image_dir) #return all the files under this path
        X = []; y = []
        for x in files:
            if x[:-4] in id_to_category:  #all elements in the x(string, file name) except for the last 4, i.e image ID
                X.append(x)
                y.append(int(id_to_category[x[:-4]]))
                
        # make labels in y to be one-hot vectors, create the catagory result in pattern of CNN's output
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y) 
        le_name_mapping = dict(zip(le.transform(le.classes_),le.classes_))

        print('len of X: {}, # of categories: {}'.format(len(X), max(y) + 1))

        # split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        return X_train, X_test, y_train, y_test, max(y) + 1, le_name_mapping


class DataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, dataset, dataset_size, params):
        self.batch_size = params['batch_size']
        self.shuffle = params['shuffle']
        self.n_classes = params['n_classes']
        self.X, self.y, self.transform = dataset
        self.image_dir = osp.join(Config['root_path'], 'images')
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X)/self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index+1) * self.batch_size]
        X, y = self.__data_generation(indexes)
        X, y = np.stack(X), np.stack(y)
        return np.moveaxis(X, 1, 3), tensorflow.keras.utils.to_categorical(y, num_classes=self.n_classes)

    # Generates data containing batch_size samples
    def __data_generation(self, indexes):
        X = []; y = []
        for idx in indexes:
            file_path = osp.join(self.image_dir, self.X[idx])
            X.append(self.transform(Image.open(file_path)))
            y.append(self.y[idx])
        return X, y

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
    
    # Denotes the number of batches per epoch
    def __len__(self):
        return int(np.floor(len(self.X)/self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index+1) * self.batch_size]
        X = self.__data_generation(indexes)
        X = np.stack(X)
        return np.moveaxis(X, 1, 3)

    # Generates data containing batch_size samples
    def __data_generation(self, indexes):
        X = []
        for idx in indexes:
            file_path = osp.join(self.image_dir, self.X[idx])
            X.append(self.transform(Image.open(file_path)))
        return X

    # Updates indexes, this method will be ran after each epoch
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)