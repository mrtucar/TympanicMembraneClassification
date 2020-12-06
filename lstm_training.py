import tensorflow as tf
from gevent.resolver.cares import result

print(tf.__version__)

from __future__ import print_function
import tensorflow.keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import scipy.misc
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
from keras.utils import to_categorical
from keras import backend as K
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from skimage.transform import resize
from keras.models import Sequential
import os
from sklearn.externals import joblib
from keras.layers import Flatten,Bidirectional,TimeDistributed
from keras.models import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.applications import resnet50
from keras.applications import inception_v3
from skimage import io
from keras.models import model_from_json
from keras.models import load_model
import h5py
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input, LSTM, Dense,Bidirectional 
from keras.models import Model
from sklearn.metrics import  confusion_matrix,classification_report
import pickle

"""
0 --> Chronic_otitis_media
1 --> Earwax_plug
2 --> Myringosclerosis
3 --> Normal
"""
def CreateDataset(colorSpace):
  if colorSpace == "RGB":
    lstm_Train_X = np.load("data/Feature_RGB_Train/X_OD_HP_and_SURFs.npy")
    lstm_Train_Y = np.load("data/Feature_RGB_Train/y_OD_HP_and_SURFs.npy")
    lstm_Test_X = np.load("data/Feature_RGB_Test/X_OD_HP_and_SURFs.npy")
    lstm_Test_Y = np.load("data/Feature_RGB_Test/y_OD_HP_and_SURFs.npy")
                          
    lstm_Train_X = lstm_Train_X[:,:,:1472]
    lstm_Test_X = lstm_Test_X[:,:,:1472]
    lstm_Train_Y = to_categorical(lstm_Train_Y)
    lstm_Test_Y = to_categorical(lstm_Test_Y)
    return lstm_Train_X,lstm_Test_X,lstm_Train_Y,lstm_Test_Y
  elif colorSpace == "HSV":
    lstm_Train_X = np.load("data/Feature_HSV_Train/X_OD_HP_and_SURFs.npy",allow_pickle=True)
    lstm_Train_Y = np.load("data/Feature_HSV_Train/y_OD_HP_and_SURFs.npy",allow_pickle=True)
    lstm_Test_X = np.load("data/Feature_HSV_Test/X_OD_HP_and_SURFs.npy",allow_pickle=True)
    lstm_Test_Y = np.load("data/Feature_HSV_Test/y_OD_HP_and_SURFs.npy",allow_pickle=True)
    lstm_Train_Y = to_categorical(lstm_Train_Y)
    lstm_Test_Y = to_categorical(lstm_Test_Y)

    x_train = []
    for x in lstm_Train_X:
        tmp = np.zeros((377,1472))
        tmp[:x.shape[0],:1472] = x[:,:1472]
        x_train.append(tmp)
    lstm_Train_X = np.array(x_train)

    x_test = []
    for x in lstm_Test_X:
        tmp = np.zeros((377,1472))
        tmp[:x.shape[0],:1472] = x[:,:1472]
        x_test.append(tmp)
    lstm_Test_X = np.array(x_test)

    return lstm_Train_X,lstm_Test_X,lstm_Train_Y,lstm_Test_Y
  elif colorSpace == "HED":
    lstm_Train_X = np.load("data/Feature_HED_Train/X_OD_HP_and_SURFs.npy",allow_pickle=True)
    lstm_Train_Y = np.load("data/Feature_HED_Train/y_OD_HP_and_SURFs.npy",allow_pickle=True)
    lstm_Test_X = np.load("data/Feature_HED_Test/X_OD_HP_and_SURFs.npy",allow_pickle=True)
    lstm_Test_Y = np.load("data/Feature_HED_Test/y_OD_HP_and_SURFs.npy",allow_pickle=True)
    
    lstm_Train_Y = to_categorical(lstm_Train_Y)
    lstm_Test_Y = to_categorical(lstm_Test_Y)

    x_train = []
    for x in lstm_Train_X:
        tmp = np.zeros((377,1472))
        tmp[:x.shape[0],:1472] = x[:,:1472]
        x_train.append(tmp)
    lstm_Train_X = np.array(x_train)

    x_test = []
    for x in lstm_Test_X:
        tmp = np.zeros((377,1472))
        tmp[:x.shape[0],:1472] = x[:,:1472]
        x_test.append(tmp)
    lstm_Test_X = np.array(x_test)

    return lstm_Train_X,lstm_Test_X,lstm_Train_Y,lstm_Test_Y

def MinMaxNormalization(x_train,x_test):
  scaler = MinMaxScaler()
  x_train  = x_train.reshape(-1,554944) #555321
  scaler.fit(x_train)
  x_train = scaler.transform(x_train)
  x_train  = x_train.reshape(-1,377,1472)

  scaler = MinMaxScaler()
  x_test  = x_test.reshape(-1,554944) #555321
  scaler.fit(x_test)
  x_test = scaler.transform(x_test)
  x_test  = x_test.reshape(-1,377,1472)
  return x_train,x_test

def classic_sequence_classifier(seq_length,feature_count,class_count,rnn_width,dropOut,recurentDropOut):
  input = Input(shape=(None, feature_count))
  x = Bidirectional (LSTM(rnn_width, return_sequences=True,dropout=dropOut, recurrent_dropout=recurentDropOut))(input)
  x = Bidirectional (LSTM(rnn_width, return_sequences=True,dropout=dropOut, recurrent_dropout=recurentDropOut))(x)
  x = Bidirectional (LSTM(rnn_width,dropout=dropOut, recurrent_dropout=recurentDropOut))(x)
  x = Dense(class_count, activation='sigmoid')(x)
  return Model(input, x)

if __name__ == '__main__':
    resultFolder = "LSTM_RGB"
    os.chdir(resultFolder)
    
    rnnWidthList = [16,32,64]
    dropOutList = [0.0,0.1,0.2,0.3,0.4,0.5]
    recurentDropOutList = [0.0,0.1,0.2,0.3,0.4,0.5]
    
    x_train,x_test,y_train,y_test = CreateDataset("RGB")
    x_train,x_test = MinMaxNormalization(np.array(x_train),np.array(x_test))
    
    x_train,x_validation,y_train,y_validation = x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2,random_state=21,stratify=y_array)
    
    for rnnWidth in rnnWidthList:
      for pdropOut in dropOutList:
        for precurentDropOut in recurentDropOutList:
          subFolder = "RNN_WIDTH_"
          subFolder = subFolder+str(rnnWidth)+"_"+str(pdropOut).replace(".","")+"_"+str(precurentDropOut).replace(".","")
          
          if (os.path.exists(os.path.join(resultFolder,subFolder,"predsBest.npy"))):
            continue
    
          os.mkdir(os.path.join(resultFolder,subFolder))
          checkpoint = ModelCheckpoint(os.path.join(resultFolder,subFolder,"modelBest.hdf5"), monitor='val_loss', verbose=1,save_best_only=True, mode='auto', period=1)
          model = classic_sequence_classifier(seq_length=None,
                                              feature_count=1472,
                                              class_count=4,
                                              rnn_width=rnnWidth,
                                              dropOut = pdropOut,
                                              recurentDropOut =precurentDropOut )
          adam = Adam(learning_rate=0.0005)
          model.compile(
                        optimizer=adam,
                        loss='mse',metrics=['accuracy'])
    
          history = model.fit(x_train, y_train, epochs=150,verbose=1,batch_size=144,
                    validation_data=(x_validation,y_validation),
                    callbacks=[checkpoint]
                    )
          
          model.save(os.path.join(resultFolder,subFolder,"model.hdf5"))
          
          y_head = model.predict(x_test)
          clfReport = classification_report(np.argmax(y_test,axis=1) , np.argmax(y_head,axis=1) ,digits=4)
          cnfMatris = confusion_matrix(np.argmax(y_test,axis=1), np.argmax(y_head,axis=1))
          print(cnfMatris)
          print(clfReport)
    
          np.save(os.path.join(resultFolder,subFolder,"confMatris.npy"),cnfMatris)
          with open(os.path.join(resultFolder,subFolder,'trainHistoryDict'), 'wb') as file_pi:
                  pickle.dump(history.history, file_pi)
          with open(os.path.join(resultFolder,subFolder,'clfReport'), 'wb') as file_pi:
                  pickle.dump(clfReport, file_pi)        
          np.save(os.path.join(resultFolder,subFolder,"preds.npy"),y_head)
          np.save(os.path.join(resultFolder,subFolder,"y_test.npy"),y_test)
    
          #x_validation,y_validation
          model = load_model(os.path.join(resultFolder,subFolder,"modelBest.hdf5"))
          y_head = model.predict(x_test)
          clfReport = classification_report(np.argmax(y_test,axis=1) , np.argmax(y_head,axis=1) ,digits=4)
          cnfMatris = confusion_matrix(np.argmax(y_test,axis=1), np.argmax(y_head,axis=1))
          print(cnfMatris)
          print(clfReport)
    
          np.save(os.path.join(resultFolder,subFolder,"confMatrisBest.npy"),cnfMatris)
          with open(os.path.join(resultFolder,subFolder,'clfReportBest'), 'wb') as file_pi:
                  pickle.dump(clfReport, file_pi)        
          np.save(os.path.join(resultFolder,subFolder,"predsBest.npy"),y_head)
          np.save(os.path.join(resultFolder,subFolder,"y_test.npy"),y_test)