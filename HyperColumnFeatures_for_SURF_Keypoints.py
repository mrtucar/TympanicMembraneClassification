# -*- coding: utf-8 -*-
#%%
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import scipy.misc
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
from keras import backend as K
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from skimage.transform import resize
import os
from sklearn.externals import joblib
from keras.layers import Flatten
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
#%%
def extract_hypercolumn(model, layer_indexes, instance):
    layers = [model.layers[li].output for li in layer_indexes]
    get_feature = K.function([model.layers[0].input],layers)
    #for layer in layers:
    #    print(layer.name)

    feature_maps = get_feature(instance)
    hypercolumns = []
    
    for convmap in feature_maps:

        if len(convmap.shape)>2:
            #print(convmap.shape)
            conv_out = convmap[0, :, :, :]
            feat_map = conv_out.transpose((2,0,1))
            #print("F sz :",feat_map.shape)
            for fmap in feat_map:     
                upscaled =resize(fmap, (224, 224), mode='constant', preserve_range=True)
                hypercolumns.append(upscaled)
        else:
            for i in range(len(convmap[0])):
                upscaled = np.full((224,224), convmap[0,i])
                hypercolumns.append(upscaled)
    return np.asarray(hypercolumns)

#%%
def func_trained_on_Datos_VGG():
    model_filepath = './modelVGG/network.h5'
    
    model = tf.keras.models.load_model(
        model_filepath,
        custom_objects=None,
        compile=False
    )
    
        
    return model
#%%

def getHPs(features,kpoints):    
    
    NewHPsFull=[]
    for layer in features:
        NewHPs=[]
        for kp in kpoints:
            x,y=kp
            NewHPs.append(layer[x,y])
        NewHPsFull.append(NewHPs)
        
    NewHPsFull=np.array(NewHPsFull)
    return NewHPsFull


#%%
if __name__ == '__main__':
    model = VGG16(weights='imagenet')
    layers_extract = [2,5,9,13,17]
    
    X,y=[],[]
    
    mainFolder =  '../HED_HSV/HSV/HSV_images/'
    subFolders=os.listdir(mainFolder)
    for index,folder in enumerate(subFolders):
        
        output=index
        path2=mainFolder+folder
        resimListesi =  os.listdir(path2)    
        fineNames = []
        
        print(len(resimListesi))
        
        for index, imgRead in enumerate(resimListesi) :

            fileNames.append(path2+'/'+imgRead)
            img = image.load_img(os.path.join(path2,imgRead), target_size=(224, 224,3))
            image2=io.imread(path2+'/'+imgRead)
            print ("Bilgi:",path2+'/'+imgRead, image2.shape)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            hc2 = extract_hypercolumn(model, layers_extract, [x])        
    
            keypoint_path='keypoints/'+folder+'/'
            kpoints=joblib.load(keypoint_path+imgRead+'.pkl')
            
            NewHPsFull=getHPs(hc2,kpoints)
            print (index,imgRead,output,"raw hc2 Normal features:",hc2.shape,np.array(NewHPsFull).shape)
            
            X.append(NewHPsFull)
            y.append(output)
            
    Xnew = []
    for i in range(len(X)):
        Xnew.append(list(X[i]))
        
    np.save("/data/Feature_HSV_Train/OD_FileNames.npy",np.array(fileNames))
    np.save("/data/Feature_HSV_Train/X_OD_HP_and_SURFs.npy",Xnew)
    np.save("/data/Feature_HSV_Train/y_OD_HP_and_SURFs.npy",y)
