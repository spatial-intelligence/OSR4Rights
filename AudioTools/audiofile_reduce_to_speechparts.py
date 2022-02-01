#!/usr/bin/python3
import sys
import os
import argparse
import glob
import time
import torch
torch.set_num_threads(1)

import shutil
import torchaudio

from pprint import pprint

################################################
#Setup
#===============================================
# sudo apt install python3-pip -y
# sudo apt-get install libsndfile1
# sudo apt-get install libsndfile1-dev
# pip install torchaudio -f https://download.pytorch.org/whl/torch_stable.html
# pip install  soundfile
#################################################

#get the arguments for input and output folders

parser = argparse.ArgumentParser(description='AudioTools Service')
parser.add_argument('-i','--inputfolder', help='Input Folder', required=True)
parser.add_argument('-j','--jobid', help='Job ID', required=True)
args = vars(parser.parse_args())

#for deubigging purposes
#args={}
#args['inputfolder']='/home/dave/'
#args['jobid']=124


#added by default based on input folder
args['outputfolder']= args['inputfolder'] + 'results/'


#Load model

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)

(get_speech_timestamps,save_audio, read_audio,utils,collect_chunks) =  utils



def getSpeechOnly (fin):

    wav = read_audio(fin)
    print (fin)

    # get speech timestamps from full audio file
    speech_timestamps = get_speech_timestamps(wav, model)
    pprint(speech_timestamps)

    fout= os.path.dirname(fin)+'/results/'+os.path.basename(fin).split('.')[0]+'__voiceparts_only.wav'

    # Check if speech_timestamps array (or sequence?) contains something because save_audio will error otherwise
    # can happen when there is no speech eg music only
    if speech_timestamps:
       save_audio(fout, collect_chunks(speech_timestamps, wav), 16000) 

    print ('voice parts filtered')

    return fout



def zipit(dir_name):
  
    outfn = args['inputfolder']+'results_'+ str(args['jobid']) #don't put in the output folder otherwise get an infinte loop!
    shutil.make_archive(outfn, 'zip', dir_name)
    
    print ('---------- FINISHED ----------')





def main():
    ts = time.time()

    # make results directory
    outpath = args['outputfolder']

    if os.path.isdir(outpath)== False:
        os.mkdir(outpath)


    ##Get list of WAV files to processes
    wavlist= glob.glob(args['inputfolder']+'*.wav')+  glob.glob(args['inputfolder']+'*.WAV')
    print ('files to process:',wavlist)

    for fn_inputwav in wavlist:

       # try:
        print (" ---> Processing file: ",fn_inputwav)

            #reduce file to speech only sections
        print ('  [Job 1]: reduce to speech only sections')
        fn_sp_enhanced = getSpeechOnly(fn_inputwav)


        #except:
        #    print ("   ======>>>>  Error processing:",fn_inputwav)



    #Zip up the results
    print ('zipping results')
    zipit(args['outputfolder'])

    te = time.time()
    tt = te-ts
    print('time taken: %2.2f secs' % tt)
    print ('done')



if __name__ == "__main__":

    print ('====>>>    Running Audio Tools Services  <<<=====')
    main()

