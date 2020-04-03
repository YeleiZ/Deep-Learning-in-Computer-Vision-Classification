from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications.mobilenet import MobileNet
from data import polyvore_dataset, DataGenerator, PredictDataGenerator
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from utils import Config
import tensorflow as tf
import numpy as np


if __name__=='__main__':

    # data generators
    dataset = polyvore_dataset()
    transforms = dataset.get_data_transforms()
    X_train, X_test, y_train, y_test, n_classes,le_dictionary = dataset.create_dataset()
    if Config['debug']:
        train_set = (X_train[:100], y_train[:100], transforms['train'])
        test_set = (X_test[:100], y_test[:100], transforms['test'])
        dataset_size = {'train': 100, 'test': 100}
    else:
        train_set = (X_train, y_train, transforms['train'])
        test_set = (X_test, y_test, transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}
    params = {'batch_size': Config['batch_size'],
              'n_classes': n_classes,
              'shuffle': True
              }
    train_generator =  DataGenerator(train_set, dataset_size, params)
    test_generator = DataGenerator(test_set, dataset_size, params)
    #print(dataset_size['train'],dataset_size['test'])


    # Build Model
    dropout_rate = 0.5
    base_model = MobileNet(layers=tf.keras.layers, weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(n_classes, activation = 'softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
       layer.trainable = False


    # define optimizers
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()


    # training
    H = model.fit(train_generator,validation_data=test_generator, epochs=Config['num_epochs'], shuffle = False)
    model.save('1_model.hdf5')
    plot_model(model, to_file='1_catagory-model.png')

    
    # plot the training loss and accuracy
    N = Config['num_epochs']
    plt.figure()
    # plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    # plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper right")
    plt.savefig('1_catagory.png')


    # create dataset for prediction
    pred_x = []
    pred_y = []
    f = open("/home/ubuntu/polyvore_outfits/test_category_hw.txt", 'r')
    for row in f:
        imageID = row.split('\n')[0] + '.jpg'
        pred_x.append(imageID)
        pred_y.append(0)
    f.close()
    params = {'batch_size': 18,
              'n_classes': n_classes,
              'shuffle': False
              }
    predSet = (pred_x, pred_y, transforms['test'])
    dataset_size = {'test': len(pred_y)}
    predict_generator = PredictDataGenerator(predSet, dataset_size, params)


    # predict data using model
    test_predict= model.predict(predict_generator)


    # write items IDs and predicted catagories into a new file
    f2 = open("1_category-finetuning.txt", "w")
    for i in range(len(pred_y)):
        inputID = pred_x[i]
        prediction = np.argmax(test_predict[i])
        output = le_dictionary.get(prediction, '<Unknown>')
        f2.write("X=%s          Predicted=%s,\n" % (inputID[:-4], output))
    