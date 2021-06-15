#!/usr/bin/python3
import sys
import os
import pickle
import time
import datetime
import pdfkit
import base64
import argparse
import facesearch as fs
import encode_images as ei
import cv2


#get the arguments for input and output folders

#parser = argparse.ArgumentParser(description='FaceSearch Service')
#parser.add_argument('-i','--inputfolder', help='Input Folder', required=True)
#parser.add_argument('-j','--jobid', help='Job ID', required=True)
#args = vars(parser.parse_args())

#for deubigging purposes
args={}
args['inputfolder']='/data/Dropbox/codebackup/FaceData/job1/'
args['jobid']=1


#added by default based on input folder
args['outputfolder']= args['inputfolder'] + 'results/'


def generateReportPDF(results):
    
    outpath = args['outputfolder']
    jobid = args['jobid']

    #print ('----results----')
    #print('Generating results report... please wait')

    #==make results directory
    if os.path.isdir(outpath)== False:
        os.mkdir(outpath)

    with open(outpath +'/matches.html', 'w') as f:
        f.writelines('<html><body><h1>::FaceSearch Results::</h1>')
        f.writelines('<h4>Note: Face difference values nearer 0 are the best matches</h4><br>')

        section=[]
        
        for res in results:

            print ('>>>  TARGET:',res[0])

            f.writelines('<h3>TARGET:</h3> <br>')
            
            f.writelines(res[0])
            
            f.writelines('<table>')
            
            
            for item in res[1]:
                
                if float(item[1])<0.25 and (1 not in section):
                    f.writelines('<tr><td><h2>very good match</h2></td></tr>')
                    section.append(1)
                elif float(item[1])<0.37 and (2 not in section):
                    section.append(2)
                    f.writelines('<tr><td><br><h2>good match</h2></td></tr>')
                elif float(item[1])<0.55 and (3 not in section):
                    section.append(3)
                    f.writelines('<tr><td><br><h2>possible match</h2></td></tr>')
                elif float(item[1])<0.6 and (4 not in section):
                    section.append(4)
                    f.writelines('<tr><td><br><h2>maybe a match</h2></td><tr>')
                elif float(item[1])<0.7 and (5 not in section):
                    section.append(5)
                    f.writelines('<tr><td><br><h2>some similarities</h2></td></tr>')

                matchscore= "%.2f" % item[1]  

                print ('  >> match:',item[0])


                fin = args['inputfolder'] +  item[0]

                #make a thumbnail of the matched input img for the report
                reducedImage = ei.imgresize(fin)

                #save as match_ in the results folder
                fn_withoutpath = os.path.basename(fin)
                resultimg = outpath + 'match_'+fn_withoutpath

                cv2.imwrite(resultimg, reducedImage)
                
                f.writelines('<tr><td>'+fn_withoutpath+'</td><td>'+str(matchscore)+'</td><td> <img src="match_' +fn_withoutpath +'" alt="possible match" width="200"  > </td></tr>' )

            f.writelines('</table> <br>')


        f.writelines('</body></html>')

    
    
    #fin= outpath+'/matches.html'
    #fout = outpath+'/report_'+str(jobid)+'.pdf'
    # print (fin,fout)
    # pdfkit.from_file(fin, fout ) 

    
    print ('---------- FINISHED ----------')


##############################################################################################################################
#   MAIN code to process files in SEARCH and TARGET folders to find face matches       -- OSR4Rights Web Service 2021
###############################################################################################################################

def main():

    #set up the  path for the functions
    fs.setPath(args['inputfolder'])

    #Scan the SEARCH and TARGET folders to build FACE ENCODING FILES
    fs.procimagescnn()

    #For EACH TARGET check the encoding files against all face encodings in the SEARCH folder
    r=fs.searchfortarget()
    #print (r)

    #Generate an output from the scan - showing which images have the closest match
    generateReportPDF(r)


if __name__ == "__main__":

    print ('====>>>    Running FaceSearch Service   <<<=====')
    main()
