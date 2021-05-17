import numpy as np
import scipy.io
from sklearn.preprocessing import StandardScaler

class Load_Data():

    train_and_test_data = scipy.io.loadmat('ReducedMINIST.mat')
    train_data = train_and_test_data["SmallTrainData"]
    test_data = train_and_test_data["SmallTestData"]

    number_of_classes = test_data.shape[0]
    train_images = []
    test_images = []
    train_classes = []
    test_classes = []

    for i in range (number_of_classes):
        train_images.append(train_data[i,0])
        test_images.append(test_data[i,0])
        train_classes.append(train_data[i,1])
        test_classes.append(test_data[i,1])

    train_images = (np.asarray(train_images))
    train_classes = np.asarray(train_classes)
    test_images = (np.asarray(test_images))
    test_classes = np.asarray(test_classes)    


    train_images_norm = []
    test_images_norm = []
    train_classes_appended = []
    test_classes_appended = []

    for i in range (number_of_classes): 
        for j in range (train_images.shape[1]):
            train_images_temp = StandardScaler().fit_transform(train_images[i,j]) 
            train_images_norm.append(train_images_temp)
            train_classes_appended.append(train_classes[i,j])
        for k in range (test_images.shape[1]):
            test_images_temp = StandardScaler().fit_transform(test_images[i,k])    
            test_images_norm.append(test_images_temp)
            test_classes_appended.append(test_classes[i,k])

    # new edit
    train_images_norm = np.asarray(train_images_norm)
    test_images_norm = np.asarray(test_images_norm)

    train_rows = number_of_classes * train_images.shape[1]
    test_rows = number_of_classes * test_images.shape[1]

    train_images_norm_flatten = np.reshape(train_images_norm,(train_rows,784))
    test_images_norm_flatten = np.reshape(test_images_norm,(test_rows,784))

    train_classes_appended = np.reshape(train_classes_appended, (train_rows,10))
    test_classes_appended = np.reshape(test_classes_appended, (test_rows,10))

    train_lables = []
    test_lables = []

    for i in range(len(test_classes_appended)):
        temp = np.argmax(test_classes_appended[i])
        test_lables.append(temp)

    for i in range(len(train_classes_appended)):
        temp = np.argmax(train_classes_appended[i])
        train_lables.append(temp)

    test_lables = np.asarray(test_lables) 
    train_lables = np.asarray(train_lables) 