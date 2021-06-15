import face_recognition
import os
import sys
import pickle
import imutils
import cv2
from sklearn.cluster import DBSCAN
import facesearch as fs
import numpy as np
import glob
import json

def writeEncodingFile(img,mode):

    #print ('encoding:',img)

    fn_withoutext = os.path.splitext(img)[0]

    print (fn_withoutext)

    if not os.path.isfile(fn_withoutext+'.fe_cnn'):

        image,facelocs,face_encodings,hashcode,numUniqueFaces=getFaceEncodings(img,2,5)

        facedetails = {}
        facedetails['fn']=image
        facedetails['face_locations']=facelocs
        facedetails['face_encodings']=face_encodings
        facedetails['filehash']=hashcode
        facedetails['numUniquefaces']=numUniqueFaces
        writeEncoding(facedetails,fn_withoutext+'.fe_cnn')

            
def drawbox(fn,img,facelocations):
    # print ('facebox:'+ fn)

    for (top, right, bottom, left) in facelocations:
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
    
    cv2.imwrite(fn, img)
    

def imgresize(imgfn):

    img= loadimg(imgfn)
    (h, w) = img.shape[:2]
    #resize image before searching
    if h>w:
        thumbnailimg=imutils.resize(img,height=200) 
        #print ('resize-h')

    else:
        thumbnailimg=imutils.resize(img,width=200) 
        #print ('resize-w')

    image_rgb = cv2.cvtColor(thumbnailimg,cv2.COLOR_BGR2RGB)

    return image_rgb



def loadimg (fnimg):
    return face_recognition.load_image_file(fnimg)

def imagehash(img,hashSize=8):
    resized = cv2.resize(img, (hashSize + 1, hashSize))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def writeEncoding(enc,pathfn):
    with open(pathfn, 'wb') as fp:
        pickle.dump(enc, fp)


def getFaceEncodings(image,times_upsample,jitters=5):
    
    fr_image= face_recognition.load_image_file(image)
    numUniqueFaces=0
    hashcode = imagehash(fr_image)
        
    (h, w) = fr_image.shape[:2]

    print (h,w)

    #resize image before searching
    if h>w:
        fr_image2=imutils.resize(fr_image,height=600) 
        #print ('resize-h')

    else:
        fr_image2=imutils.resize(fr_image,width=600) 
        #print ('resize-w')

    print ('cnn: locating faces',image)

    face_locations = face_recognition.face_locations(fr_image2, number_of_times_to_upsample=times_upsample, model="cnn")
    #drawbox(image.split('.')[0]+'_facebox_1k.'+image.split('.')[1],fr_image2,face_locations)   #save image with boxes

    face_encodings= face_recognition.face_encodings(fr_image2, face_locations,num_jitters=jitters)

    numUniqueFaces = len (face_encodings)


    print (len(face_encodings))
    print (hashcode)
    print ('=====')

    return image,face_locations,face_encodings,hashcode,numUniqueFaces
