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
import shutil


#get the arguments for input and output folders

parser = argparse.ArgumentParser(description='FaceSearch Service')
parser.add_argument('-i','--inputfolder', help='Input Folder', required=True)
parser.add_argument('-j','--jobid', help='Job ID', required=True)
args = vars(parser.parse_args())

#for deubigging purposes
#args={}
#args['inputfolder']='/data/Dropbox/codebackup/FaceData/job1/'
#args['jobid']=1


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
        f.writelines('<html>')
        f.writelines('<head><style> #results {  font-family: Arial, Helvetica, sans-serif; border-collapse: collapse;  width:60%; }')
        f.writelines('#results td, #results th { border: 1px solid #ddd padding: 8px;}')
        f.writelines('#results tr:hover {background-color: #ddd;} #results th { padding-top: 12px; padding-bottom: 12px; text-align: left; background-color: #04AA6D; color: white; } </style> </head>')
        
        
        f.writelines('<body><h1> FaceSearch Results </h1>')
        f.writelines('<h4>Note: Face difference values nearer 0 are the best matches</h4><br>')

        
        for res in results:

            section=[]

            f.writelines('<table id="results"> ')

            print ('>>>  TARGET:',res[0])

            f.writelines('<tr><td><h3>TARGET:</h3> </td>')
            f.writelines('<td>'+res[0]+'</td><td>')

            
            #make a thumbnail of the target img for the report
            fint = args['inputfolder'] +  res[0]
            reducedImage = ei.imgresize(fint)

            #save as target_ in the results folder
            fn_withoutpath = os.path.basename(fint)
            resultimg = outpath + 'target_'+fn_withoutpath

            cv2.imwrite(resultimg, reducedImage)
            f.writelines('<img src="target_' +fn_withoutpath +'" alt=target image" width="250" >  </td><tr><br></tr>' )


            
            for item in res[1]:
                
                if float(item[1])<0.25 and (1 not in section):
                    f.writelines('<tr><td style="background-color:#f2f2f2"><h3>very good match</h3></td><td style="background-color:#f2f2f2"></td><<td style="background-color:#f2f2f2"></td></tr>')
                    section.append(1)
                elif float(item[1])<0.37 and (2 not in section):
                    section.append(2)
                    f.writelines('<tr><td style="background-color:#f2f2f2"><h3>good match</h3></td><td style="background-color:#f2f2f2"></td><<td style="background-color:#f2f2f2"></td></tr>')
                elif float(item[1])<0.55 and (3 not in section):
                    section.append(3)
                    f.writelines('<tr><td style="background-color:#f2f2f2"><h3>possible match</h3></td><td style="background-color:#f2f2f2"></td><td style="background-color:#f2f2f2"></td></tr>')
                elif float(item[1])<0.6 and (4 not in section):
                    section.append(4)
                    f.writelines('<tr><td style="background-color:#f2f2f2"><h3>maybe a match</h3></td><td style="background-color:#f2f2f2"></td><td style="background-color:#f2f2f2"></td></tr>')
                elif float(item[1])<0.7 and (5 not in section):
                    section.append(5)
                    f.writelines('<tr><td style="background-color:#f2f2f2"><h3>some similarities</h3></td><td style="background-color:#f2f2f2"></td><td style="background-color:#f2f2f2"></td></tr>')

                matchscore= "%.2f" % item[1]  

                print ('  >> match:',item[0])


                fin = args['inputfolder'] +  item[0]

                #make a thumbnail of the matched input img for the report
                reducedImage = ei.imgresize(fin)

                #save as match_ in the results folder
                fn_withoutpath = os.path.basename(fin)
                resultimg = outpath + 'match_'+fn_withoutpath

                cv2.imwrite(resultimg, reducedImage)
                
                f.writelines('<tr><td>'+fn_withoutpath+'</td><td>'+str(matchscore)+'</td><td> <img src="match_' +fn_withoutpath +'" alt="possible match" width="250"  > </td></tr>' )
            f.writelines('</table> <hr> <br>')

        f.writelines('</body></html>')



def zipit(dir_name):
    outfn = args['inputfolder']+'results_'+str(args['jobid'])  #don't put in the output folder otherwise get an infinte loop!
    shutil.make_archive(outfn, 'zip', dir_name)
    
    
    #fin= outpath+'/matches.html'
    #fout = outpath+'/report_'+str(jobid)+'.pdf'
    # print (fin,fout)
    # pdfkit.from_file(fin, fout ) 

    
    print ('---------- FINISHED ----------')


##############################################################################################################################
#   MAIN code to process files in SEARCH and TARGET folders to find face matches       -- OSR4Rights Web Service 2021
###############################################################################################################################

def main():
    ts = time.time()

    #set up the  path for the functions
    fs.setPath(args['inputfolder'])

    #Scan the SEARCH and TARGET folders to build FACE ENCODING FILES
    print ('encoding faces')
    fs.procimagescnn()

    #For EACH TARGET check the encoding files against all face encodings in the SEARCH folder
    print ('matching faces')
    r=fs.searchfortarget()
    #print (r)

    #Generate an output from the scan - showing which images have the closest match
    print ('generating report')
    generateReportPDF(r)

    #Zip the html report into a single zip file
    print ('zipping report')
    zipit(args['outputfolder'])

    te = time.time()
    tt = te-ts
    print('time taken: %2.2f secs' % tt)
    print ('done')


if __name__ == "__main__":

    print ('====>>>    Running FaceSearch Service   <<<=====')
    main()
