#!/usr/bin/python3
import sys
import os
import argparse
import glob
import time
import torch
import shutil
torch.set_num_threads(1)
import torchaudio
from speechbrain.pretrained import SpectralMaskEnhancement
from speechbrain.pretrained import SepformerSeparation as separator
from speechbrain.dataio.dataio import write_audio
from pprint import pprint
import speechbrain as sb
from speechbrain.dataio.dataio import read_audio
from speechbrain.pretrained import EncoderDecoderASR


################################################
#Setup
#===============================================
#
# pip install speechbrain
# pip install transformers
# pip install spleeter
# pip install -q torchaudio soundfile#
#################################################


#get the arguments for input and output folders

#parser = argparse.ArgumentParser(description='AudioTools Service')
#parser.add_argument('-i','--inputfolder', help='Input Folder', required=True)
#parser.add_argument('-j','--jobid', help='Job ID', required=True)
#args = vars(parser.parse_args())

#for deubigging purposes
args={}
args['inputfolder']='/home/dave/'
args['jobid']=124


#added by default based on input folder
args['outputfolder']= args['inputfolder'] + 'results/'


#Load models

#-----Speech only-----
#original
#model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad',  force_reload=True)

#using forked copy
model, utils = torch.hub.load(repo_or_dir='spatial-intelligence/silero-vad', model='silero_vad',  force_reload=True)

(get_speech_ts,get_speech_ts_adaptive,save_audio,read_audio,state_generator,single_audio_stream,collect_chunks) = utils

#-----getSeparation2Voices -----
modelseparate = separator.from_hparams(source="speechbrain/sepformer-wsj02mix", savedir='pretrained_models/sepformer-wsj02mix')

#-----doSpeechEnhancement  -----
enhance_model = SpectralMaskEnhancement.from_hparams(source="speechbrain/metricgan-plus-voicebank",savedir="pretrained_models/metricgan-plus-voicebank",)

# -----SpeechtoText  -----
asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="pretrained_models/asr-crdnn-rnnlm-librispeech")



def getSpeechOnly (fin):

    wav = read_audio(fin)

    # get speech timestamps from full audio file
    speech_timestamps = get_speech_ts(wav, model,  num_steps=4)
    pprint(speech_timestamps)

    fout= os.path.dirname(fin)+'/results/'+os.path.basename(fin).split('.')[0]+'__voiceparts_only.wav'

    save_audio(fout, collect_chunks(speech_timestamps, wav), 16000) 

    print ('done voice parts only')

    return fout


def getSeparation2Voices(fin):

    est_sources = modelseparate.separate_file(path=fin) 

    w1 = est_sources[:, :, 0].detach().cpu().squeeze()
    w2 = est_sources[:, :, 1].detach().cpu().squeeze()

    fout= os.path.dirname(fin)+'/results/'+os.path.basename(fin).split('.')[0]

    write_audio(fout+'__separate1.wav',w1,8000)
    write_audio(fout+'__separate2.wav',w2,8000)

    print ('done separate 2 voices')




def doSpeechEnhancement(fin,fout):

    # Load and add fake batch dimension
    noisy = enhance_model.load_audio(fin).unsqueeze(0)
    # Add relative length tensor
    enhanced = enhance_model.enhance_batch(noisy, lengths=torch.tensor([1.]))
    # Saving enhanced signal on disk
    torchaudio.save(fout, enhanced.cpu(), 16000)

    print ('done enhacement')



def doSpeechtoText(fin):

    fout = os.path.splitext(fin)[0]

    textout = asr_model.transcribe_file(fin)
    print (textout)

    with open(fout+"__transcript.txt", "w") as text_file:
        text_file.write(textout)

    
    #try on enhanced speech only version too
    textout2 = asr_model.transcribe_file(fin)
    print (textout2)

    with open(fout+"__transcript2.txt", "w") as text_file:
        text_file.write(textout2)

    print ('done transcript')




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

    for fn_inputwav in wavlist:

        try:

            print (" ---> Processing file: ",fn_inputwav)

            fn_withoutext = fn_inputwav

            #Job 1 - reduce file to speech only sections
            print ('  [Job 1]: reduce to speech only sections')
            fn_sp_enhanced = getSpeechOnly(fn_inputwav)

            #Job 2 - separate 2 mixed voice channels into individual voice parts by speaker
            print ('   [Job 2]: separate voices')
            getSeparation2Voices(fn_inputwav)

            #Job 3 - speech enhancement
            print ('   [Job 3a]: enhance all')
            fout1= os.path.dirname(fn_inputwav)+'/results/'+os.path.basename(fn_inputwav).split('.')[0]+'enhanced_all.wav'
            doSpeechEnhancement(fn_inputwav,fout1)

            print ('   [Job 3b]: enhance voiceparts only')
            fout2= os.path.dirname(fn_inputwav)+'/results/'+os.path.basename(fn_inputwav).split('.')[0]+'__enhanced_voiceparts_only.wav'
            doSpeechEnhancement(fn_sp_enhanced, fout2)

            #Job 4 - speech to text
            print ('   [Job 4]: speech to text')
            #doSpeechtoText(fn_inputwav)


        except:
            print ("   ======>>>>  Error processing:",fn_inputwav)



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

