import numpy as np
import scipy as sp
import scipy.misc
import cv2,os,shutil
from skimage import io
from sklearn.externals import joblib
import matplotlib
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import io
from random import randint
import argparse

def isCheck_folder(path):
    if not os.path.isdir(path):
        os.mkdir( path);

def main(dataset_name,color_spaces):    
    
    for colorSpace in color_spaces:

        path1=dataset_name
        
        subFolders=os.listdir(path1)
        
        for folder in subFolders:
            imagelist=os.listdir(path1+folder)
            keypoint_path='./keypoints/'+folder+'/'
            isCheck_folder(keypoint_path)
            
            save_dir="./keypoints_on_earDrum_images/"+folder+'/'
            isCheck_folder(save_dir)

            for retina_name in imagelist:      
            
                path2=path1+folder+'/'+retina_name
                
                image=io.imread(path2)
                
                image_resized = image#resize(image, (224,224,3))
                io.imsave(path2,image_resized)
                       
            
                img = cv2.imread(path2)
         
                keypoint_list=[]
    
                #detector = cv2.xfeatures2d.SURF_create()
                detector = cv2.SURF(100)
                
                kpoints, desc = detector.detectAndCompute( img, None )  
    
                if len(kpoints)!=0:        
                    for m in range(0,len(kpoints)):
                        x,y=int(kpoints[m].pt[0]),int(kpoints[m].pt[1])
    
                        #x,y = i.ravel()
                        cv2.circle(img,(x,y),1,255,0,0)
                        
                        keypoint_list.append([int(kpoints[m].pt[1]),int(kpoints[m].pt[0])])
                        
                print ("Color Space:",colorSpace," ",retina_name," point count:",len(keypoint_list))
                cv2.imwrite(save_dir+retina_name,img)                

def point_sayisini_goster(dataset_name,color_spaces,kpoints_nameList):
    
        en_az_sayisi=None
        
        for colorSpace in color_spaces:
            
            for kp_algorithm in kpoints_nameList:
                
                path1=dataset_name
                subFolders=os.listdir(path1)
                
                for folder in subFolders:
                
                    imagelist=os.listdir(path1+folder)

                    for retina_name in imagelist:
                        
                        keypoint_path='./keypoints/'+folder+'/'
                                     
                        kpoints=joblib.load(keypoint_path+retina_name+'.pkl')
                        
                        if en_az_sayisi==None:
                            en_az_sayisi=len(kpoints)
                        else:
                            if len(kpoints)<en_az_sayisi:
                                en_az_sayisi=len(kpoints)
                        
                        
                        print (dataset_name," Dataset", retina_name,kp_algorithm,"- Keypoint count:",len(kpoints))

                            
                    print ("*************************")
                print ("*************************")
                print ("*************************")
        print (en_az_sayisi)

if __name__ == '__main__':
    dataset_name='./earDrumData_original/'
    color_spaces=['rgb']
    kpoints_nameList=['SURF']
    main(dataset_name,color_spaces)




