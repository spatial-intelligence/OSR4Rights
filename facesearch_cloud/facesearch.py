import glob
import os
import sys
from PIL import Image, ImageDraw
import encode_images as ei
import findtarget as ft
import time
from pandas import DataFrame
import numpy as np
import json

#Setup some Global Variable for the Image Path for Images to Search and Target faces to look for
path=''
search_imgpath=''
target_imgpath=''

#Set the search paths
def setPath(p):

    global path, search_imgpath,target_imgpath 
    path = p
    search_imgpath = p + 'search/'
    target_imgpath = p + 'target/'    
    #print ('Path set:',path)


#Check any existing enccoding files hve the same image hash as the corresponding image of the same name
def checkimages_against_hashes():
    filechanges = []
    filecount=0

   # try:

    for img in glob.glob(search_imgpath+'*.jpg')+  glob.glob(search_imgpath+'*.jpeg')+  glob.glob(search_imgpath+'*.JPEG')+glob.glob(search_imgpath+'*.png')+glob.glob(search_imgpath+'*.JPG')+ glob.glob(search_imgpath+'*.PNG'):

        cnnfiles = (glob.glob(search_imgpath+'*.fe_cnn'))
        cfiles = [w.replace('.fe_cnn', '') for w in cnnfiles]

        if (os.path.splitext(img)[0] in cfiles):   #-- need to scan just first part of filename against first part in the list 
            fnhash= ft.readEncoding(os.path.splitext(img)[0] +'.fe_cnn')['filehash']   ##ft

            #print (filecount,fnhash)

            filecount += 1
            hashimg= ei.imagehash(ei.loadimg(img)) ## ei 
            if hashimg != fnhash:
                filechanges.append(img)
    #except:
    #    print ('Error with a face encoding file')

    return filechanges,filecount



def filescan():
   
    cnnfacesfound = []

    for img in (glob.glob(search_imgpath+'*.fe_cnn')):

            facecount= ft.readEncoding(img)['numUniquefaces']  ## ft.
            cnnfacesfound.append([facecount,img])

    return cnnfacesfound



def searchfiles():
    #print ('search path:',search_imgpath)

    fecnn = len (glob.glob(search_imgpath+'*.fe_cnn'))
    return glob.glob(search_imgpath+'*.jpg')+ glob.glob(search_imgpath+'*.jpeg')+ glob.glob(search_imgpath+'*.JPEG')+ glob.glob(search_imgpath+'*.png')+glob.glob(search_imgpath+'*.JPG')+ glob.glob(search_imgpath+'*.PNG'),fecnn


def searchfiles_encoding_cnn():
    fecnn = (glob.glob(search_imgpath+'*.fe_cnn'))
    return fecnn

def targetfiles():
    fecnn = len (glob.glob(target_imgpath+'*.fe_cnn'))
    return glob.glob(target_imgpath+'*.jpg')+glob.glob(target_imgpath+'*.jpeg')+glob.glob(target_imgpath+'*.JPEG')+ glob.glob(target_imgpath+'*.png')+glob.glob(target_imgpath+'*.JPG')+ glob.glob(target_imgpath+'*.PNG'),fecnn


def targetfiles_encoding_cnn():
    fecnn = (glob.glob(target_imgpath+'*.fe_cnn'))
    return fecnn


def removeEncodingFileCNN(fns):
    #error in encoding file - so delete and rebuild before scan
    
    #print (fns)
    
    for fn in fns:
        #print (fn)
        fn_withoutext = os.path.splitext(fn)[0]
        fn_fe = fn_withoutext+'.fe_cnn'
        os.remove (fn_fe)
        #print ('Removed FE',fn_fe)



def procimagescnn():
    sfiles,sfecnn = searchfiles()
    tfiles,tfecnn  = targetfiles()

    #calc search images
    for img in sfiles:
        #print (img)
        ei.writeEncodingFile(img,'cnn')  ##ei.

    #now for target images
    for img in tfiles:
        #print (img)
        ei.writeEncodingFile(img,'cnn')  ##ei.
       



def searchfortarget():
    #check encoding files against image hashes
    #Any differences - delete encoding and re-build
    fns,filecount = checkimages_against_hashes()
    removeEncodingFileCNN(fns)
        
    #ensure updated - CNN
    procimagescnn()
    
    #run hashcheck on the images against any previous encodings (check files tab)
    sfiles,sfecnn = searchfiles()  
    tfiles,tfecnn = targetfiles()

    
    #load all the searchfile face encodings
    senc = searchfiles_encoding_cnn()
    ft.buildfaceDB(senc)  ##ft.

    #load all the target face encodings and do comparison
    tenc = targetfiles_encoding_cnn()
    res = ft.dosearch(tenc) ##ft.

     #unique list of matches (images) - showing best score match per image
    df = DataFrame (res,columns=['fn','diff'])

    uniqueresults = []
    for index, row in df.iterrows():
        r = (row['diff'])
        minscores={}
        for i in r:
            f=i[0].replace(path,'')
            if f in minscores and minscores[f] > i[1]:
                minscores[f]=i[1]
            if f not in minscores:
                minscores[f]=i[1]
        uniqueresults.append([row['fn'].replace(path,''), minscores.items() ])

    #uniqueresults=df.values.tolist()

    return uniqueresults



'''
Scans the .fe file to check if the image hash in the .fe matches that for the corresponding image (based on img filename)
Returns the list of files which don't match, and count of number of images scanned

'''
def checkfilehash():
    #check number of files
    
    #check if file has fe_* that matches the image hash
    filechanges=[]

    #print ('running file hash check')
    filechanges,filecount = checkimages_against_hashes()
    return filechanges, filecount 



