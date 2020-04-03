from tensorflow.keras.layers import MaxPooling2D, Dense, Dropout, Input, Conv2D, Flatten
from data import polyvore_dataset, DataGenerator, PredictDataGenerator
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from utils import Config
import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
from sklearn.preprocessing import LabelEncoder



if __name__=='__main__':

    # data generators
    dataset = polyvore_dataset()
    transforms = dataset.get_data_transforms()
    X_train, X_test, y_train, y_test, n_classes, le_dictionary = dataset.create_dataset()
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
    print(dataset_size['train'],dataset_size['test'])
    # print(np.array(X_train).shape) = (200806,)
    # train_generator.shape = (70,224,224,3)


    # Build Model
    dropout_rate = 0.5
    reg_val = 0.0001
    # Block 1
    nnet_in = Input(shape=(224,224,3), name='nnet_in')
    x1 = Conv2D(8,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val))(nnet_in)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)
    # Block 2
    x2 = Conv2D(16,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val))(x1)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    # Block 3
    x3 = Conv2D(32,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val))(x2)
    x3 = MaxPooling2D(pool_size=(2, 2))(x3)
    x3 = Flatten()(x3)
    # Fully connected
    x4 = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val))(x3)
    x4 = Dense(256, activation='tanh', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val))(x4)
    x4 = Dropout(dropout_rate)(x4)
    nnet_out = Dense(n_classes, activation = 'softmax',kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001))(x4)
    
    model = Model(inputs=nnet_in, outputs=nnet_out)

    # define optimizers
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # training
    H = model.fit(train_generator,validation_data=test_generator, epochs=Config['num_epochs'], shuffle = False)
    model.save('model_from_scratch.hdf5')
    plot_model(model, to_file='catagory-model-from-scratch.png')

    
    # plot the training loss and accuracy
    N = Config['num_epochs']
    plt.figure()
    # plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    # plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training and Validation Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper right")
    plt.savefig('catagory-from-scratch.png')



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
    f2 = open("category.txt", "w")
    for i in range(len(pred_y)):
        inputID = pred_x[i]
        prediction = np.argmax(test_predict[i])
        output = le_dictionary.get(prediction, '<Unknown>')
        f2.write("X=%s          Predicted=%s,\n" % (inputID[:-4], output))
    