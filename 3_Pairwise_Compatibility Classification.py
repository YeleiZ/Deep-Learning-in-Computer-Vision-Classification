from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate, Lambda
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.mobilenet import MobileNet
import matplotlib.pyplot as plt
from utils import Config
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from data2 import polyvore_dataset, DataGenerator, PredictDataGenerator


if __name__=='__main__':

    # data generators
    dataset = polyvore_dataset()
    transforms = dataset.get_data_transforms()
    X_train, X_test, y_train, y_test, n_classes = dataset.create_dataset()
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

    # Build model
    in_nn = Input((224,224,6))
    split1 = Lambda(lambda x: x[:, :, :, :3])(in_nn)
    split2 = Lambda(lambda x: x[:, :, :, 3:])(in_nn)
    # 1st branch operates on the first input
    x = Conv2D(8,(3,3), padding='same', activation='relu')(split1)
    x = Conv2D(16,(3,3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32,(3,3), padding='same', activation='relu')(x)
    x = Conv2D(64,(3,3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128,(3,3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # 2nd branch opreates on the second input
    y = Conv2D(8,(3,3), padding='same', activation='relu')(split2)  
    y = Conv2D(16,(3,3), padding='same', activation='relu')(y)
    y = MaxPooling2D(pool_size=(2, 2))(y)
    y = Conv2D(32,(3,3), padding='same', activation='relu')(y)
    y = Conv2D(64,(3,3), padding='same', activation='relu')(y)
    y = MaxPooling2D(pool_size=(2, 2))(y)
    y = Conv2D(128,(3,3), padding='same', activation='relu')(y)
    y = MaxPooling2D(pool_size=(2, 2))(y)
    # combine the output of the two branches
    combined = concatenate([x, y])
    # apply a FC layer 
    z = Flatten()(combined)
    z = Dense(400, activation="relu")(z)
    z = Dense(200, activation="relu")(z)
    z = Dropout(0.5)(z)
    z = Dense(100, activation="tanh")(z)
    z = Dense(1, activation="sigmoid")(z)

    model = Model(inputs=in_nn, outputs=z)
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

    H = model.fit(train_generator, validation_data=test_generator, epochs=Config['num_epochs'], shuffle=False)

    model.save('3_model.hdf5')
    plot_model(model, to_file='3_compatibility-model.png')

    
    # plot the training loss and accuracy
    N = Config['num_epochs']
    plt.figure()
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training and Validation Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper right")
    plt.savefig('3_compatibility.png')



    # create dataset for prediction
    pred_x = []
    pred_y = []
    f = open("/home/ubuntu/polyvore_outfits/test_pairwise_compat_hw.txt", 'r')
    for row in f:
        imageID = row.split()[0] + '.jpg' + '   ' + row.split()[1] + '.jpg'
        pred_x.append(imageID)
        pred_y.append(0)
    f.close()

    params = {'batch_size': Config['batch_size'],
              'n_classes': n_classes,
              'shuffle': False
              }

    predSet = (pred_x, pred_y, transforms['test'])
    dataset_size = {'test': len(pred_y)}
    predict_generator = PredictDataGenerator(predSet, dataset_size, params)

    # predict data using model
    test_predict= model.predict(predict_generator)

    # write items IDs and predicted catagories into a new file
    f2 = open("3_pair-compatibility.txt", "w")
    for i in range(len(test_predict)):
        inputID1 = pred_x[i].split()[0]
        inputID2 = pred_x[i].split()[1] 
        prediction = test_predict[i].round()
        f2.write("X1=%s    X2=%s        Predicted=%s,\n" % (inputID1[:-4],inputID2[:-4], prediction))