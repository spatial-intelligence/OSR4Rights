#!/usr/bin/python3
import sys
import dlib  
import csv
import face_recognition
import os
import pickle
import time
import datetime
from PIL import Image, ImageDraw, ImageFont
import gc
import numpy
import psycopg2
import pdfkit
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.nonmultipart import MIMENonMultipart
import base64

#comment

# Database connection parameters
database = 'jobqueue'
user = 'dbuserhere'
password = 'passwordhere'
host = '127.0.0.1'
port = 5432

##########################################################
# Function to excecute SQL query and return row list
def pg_query(sql):
    try:
        conn = psycopg2.connect(host=host, port=port, database=database, user=user, password=password)
        cur = conn.cursor()
        cur.execute(sql)
        ret = cur.fetchall()
        cur.close()
        conn.close()
    except:
        exit("I am unable to connect to the database")
        ret= {}
  
    return ret

def pg_exec(sql):
    try:
        #print (sql)
        conn2 = psycopg2.connect(host=host, port=port, database=database, user=user, password=password)
        cur2 = conn2.cursor()
        cur2.execute(sql)
        conn2.commit()
        cur2.close()
        conn2.close()
    except:
        exit("I am unable to connect to the database")
    #time.sleep(1)



def updatejob(jobid):
    sql="UPDATE jobs SET job_completed=now() where jobid="+str(jobid)+";"
    pg_exec(sql)

def sendEmail (emailto,pdf):
               
    msg = MIMEMultipart()

    emailfrom ='osr4rights@gmail.com'

    msg['From'] = emailfrom
    msg['To'] = emailto
    msg['Subject'] = 'OSR4RIGHTS Job Report'
 
    fp = open(pdf, 'rb')

    attach = MIMENonMultipart('application', 'pdf')
    payload = base64.b64encode(fp.read()).decode('ascii')
    attach.set_payload(payload)
    attach['Content-Transfer-Encoding'] = 'base64'
    fp.close()

    attach.add_header('Content-Disposition', 'attachment', filename = 'report.pdf')

    body = "Please find attached report (PDF)"
    msg.attach(MIMEText(body, 'plain'))

    msg.attach(attach)

    session = smtplib.SMTP('smtp.gmail.com',587)
    session.ehlo()
    session.starttls()

    session.login('osr4rights@gmail.com', 'passwordhere')

    print ('Sending to:'+ emailto + ' at '+ str(datetime.datetime.now()))
    
    session.sendmail(emailfrom, emailto, msg.as_string())
    session.quit()
    
    print ('OK sent to:'+ emailto + ' at '+ str(datetime.datetime.now()))
    print ('-------------------------------------\n')
    
    time.sleep(5)



def process_faceimg(jobid):
    print ('processing job id:' + str(jobid))

    #face database
    faceDB={}
    targetfaceencodings=None
    targetfacelocations=None

    #get target image face encodings
    ext=('.jpg','.png','.jpeg','.gif')
    scanpath = "/workspace/webservices/job"+str(jobid)
    images_toscan=scanDirectory(scanpath,ext)

    if len(images_toscan)>0:

        #go thru the list and build face encoding file
        writeEncodingFiles(images_toscan)

        #get the target file  - called target.jpg
        targetfaceencodings= updateTargetFaceEncodings(scanpath)

        if targetfaceencodings is not None:

            #scan files to dictionary
            ext='.fe'
            felist = scanDirectory(scanpath,ext)

            for encfile in felist:
                resize,loc,enc,imgfn = readEncoding(encfile)
                faceDB[encfile]=[loc,enc,resize,imgfn]

            results=findFaceDistances(targetfaceencodings,faceDB)
            sortedresults=sorted(results.items(), key=lambda x: x[1])
            print ('----results----')
            print('Generating results report... please wait')
            print (sortedresults)

            #==make results directory
            outpath=scanpath+'/results'
            if os.path.isdir(outpath)== False:
                os.mkdir(outpath)

            with open(outpath +'/matches.html', 'w') as f:
                f.writelines('<html><body><h1>Face Search Results</h1>')
                f.writelines('<h4>Note: Face difference values nearer 0 are the best matches</h4><br><br>')
                f.writelines('<table>')
                section=[]
                for item in sortedresults:
                    if item[1][0]<0.25 and (1 not in section):
                        f.writelines('<tr><td><h2>very good match</h2></td></tr>')
                        section.append(1)
                    elif item[1][0]<0.37 and (2 not in section):
                        section.append(2)
                        f.writelines('<tr><td><br><br><h2>good match</h2></td></tr>')
                    elif item[1][0]<0.55 and (3 not in section):
                        section.append(3)
                        f.writelines('<tr><td><br><br><h2>possible match</h2></td></tr>')
                    elif item[1][0]<0.6 and (4 not in section):
                        section.append(4)
                        f.writelines('<tr><td><br><br><h2>maybe a match</h2></td><tr>')
                    elif item[1][0]<0.7 and (5 not in section):
                        section.append(5)
                        f.writelines('<tr><td><br><br><h2>some similarities</h2></td></tr>')
                        
                    matchscore= "%.2f" % item[1][0]
                    f.writelines('<tr><td>'+item[1][2]+'</td><td>'+str(matchscore)+'</td><td><a href="match_'+item[1][2]+'"> <img src="match_' +item[1][2]+ '"height="150"> </a></td></tr>' )

                f.writelines('</table></body></html>')

                
                #======IMG RESIZE ISSUE TO RESOLVE 2500 pixels or 1250 PIXELs....============================
                for f in results:
                    fnkey=f.split('.')[0]
                    imgfn = faceDB[f][3]
                    tmpfileresize=faceDB[f][2]

                    resizeimage(tmpfileresize,fnkey+os.path.splitext(imgfn)[1],'tmp_resize.jpg')
                    drawFaceBox([results[f][1]],'tmp_resize.jpg',outpath+'/match_'+imgfn)


                #sort dictionary on scores and print out
                print ('done')
                print('Finished Searching - results in subfolder')
                return True
        else:
            print ('missing target')
            return False
    else:
        print ('no images')
        return False


def findFaceDistances(targetfaceencodings,faceDB):
    matchingfiles={}
    for sFace in targetfaceencodings:
        for f in faceDB:
            face_distances = face_recognition.face_distance(faceDB[f][1], sFace)
            
            for fd in face_distances:
                if fd < 0.625:
                    ind= numpy.where(face_distances==fd)[0][0]  #which face closest match
                    facloc=faceDB[f][0][ind]
                    imgfn=faceDB[f][3]
                    if f in matchingfiles:
                        min=matchingfiles[f][0]
                        if  fd < min:
                            matchingfiles[f]=[fd,facloc,imgfn]
                    else:
                        matchingfiles[f]=[fd,facloc,imgfn]
    return matchingfiles


def writeEncodingFiles(imglist):
    counter=0
    for img in imglist:
        print ('checking:',img)
        fnfe=os.path.splitext(img)[0]+'.fe'
        counter=counter+1
        workdone= "%.1f" % (100*counter/len(imglist))
        
        print('Scanning image #' + str(counter) + ' ('+workdone+'%)')

        if not os.path.isfile(fnfe):
            try:
                print ('>> scanning for faces')
                rs=2500
                resizeimage(rs,img,'tmp_scan.jpg')
                print (img)
                locations,encodings=getFaceEncodings('tmp_scan.jpg')
                print ('saving face encodings file')
                writeEncoding(rs,locations,encodings,img)
            except:
                try:
                    print ('>> scanning for faces')
                    rs=1250
                    resizeimage(rs,img,'tmp_scan.jpg')
                    print (img)
                    locations,encodings=getFaceEncodings('tmp_scan.jpg')
                    print ('saving face encodings file')
                    writeEncoding(rs,locations,encodings,img)
                except:
                    print('error processing file:',img)


def getFaceEncodings(image):
    gc.collect()
    fr_image=face_recognition.load_image_file(image)
    face_locations = face_recognition.face_locations(fr_image, number_of_times_to_upsample=0, model="cnn")
    face_encodings = face_recognition.face_encodings(fr_image, face_locations,num_jitters=3)  
    return face_locations,face_encodings

def writeEncoding(resizevalue,loc,enc,pathfn):
    fnfe=os.path.splitext(pathfn)[0]+'.fe'
    with open(fnfe, 'wb') as fp:
        pickle.dump(resizevalue,fp)
        pickle.dump(loc, fp)
        pickle.dump(enc, fp)
        pickle.dump(pathfn,fp)

def readEncoding(pathfn):
    with open (pathfn, 'rb') as fp:
        resizevalue=pickle.load(fp)
        loc=pickle.load(fp)
        enc=pickle.load(fp)
        imgpathfn = pickle.load(fp)
        imgfn=os.path.basename(imgpathfn)
    return resizevalue,loc,enc,imgfn

def scanDirectory(path,ext):
    try:
        files=[]
        for file in os.listdir(path):
            if file.endswith(ext):
                files.append(os.path.join(path, file))          
        return files
    
    except:
        return []


def resizeimage(size,fnin,fnout):
    basewidth = size
    img = Image.open(fnin)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(fnout)

def cropFace(top,right,bottom,left):
    image = face_recognition.load_image_file("tmp_findface.jpg")
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()

def drawFaceBox(facelocations,fnin,fnout):
    img = Image.open(fnin)
    img2= img.point(lambda p: p * 0.5)
    draw = ImageDraw.Draw(img2)
    i=0
    for face in facelocations:
        i=i+1
        top, right, bottom, left = face
        draw.rectangle(((left,top),(right,bottom)),outline='white')
        draw.rectangle(((left+1,top+1),(right,bottom)),outline='white')
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 48, encoding="unic")
        draw.text((left,top), str(i),font=font)

    del draw

    img2.save(fnout)

def getFaceLocations(times_upsample):
    fr_image= face_recognition.load_image_file('tmp_findface.jpg')
    face_locations = face_recognition.face_locations(fr_image, number_of_times_to_upsample=times_upsample, model="cnn")
    return face_locations

def updateTargetFaceEncodings(scanpath):
    try:
        tarfacefe = readEncoding(scanpath+'/target.fe')
        return tarfacefe[2]
    except:
        print ('missing target face image')
        return None


def yolo9000_imageproc(img):
    cmd = '/tools/yolo9000/scanimage.sh ' + img
    stream=os.popen(cmd)
    output = stream.read()
    desc=output.split('\n')
    return desc [1:-1]

def yolo_imageproc(img):
    cmd = '/tools/yolo/scanimage.sh ' + img
    stream=os.popen(cmd)
    output = stream.read()
    desc=output.split('\n')
    return desc 


def process_yolometa_img(jobid):
    results={}
    ext=('.jpg','.png','.jpeg','.gif')
    scanpath = "/workspace/webservices/job"+str(jobid)
    images_toscan=scanDirectory(scanpath,ext)

    for img in images_toscan:
        #yolo - general
        r=yolo_imageproc(img)
        print (r)
        fnimg=os.path.basename(img)
        results[fnimg+'_g']=r
        #yolo 9000 - more specifc but can miss things
        r=yolo9000_imageproc(img)
        print (r)
        fnimg=os.path.basename(img)
        results[fnimg+'_s']=r

    return results


def main():
    while True:
        try:
            mainprocess()
            time.sleep(60)
        except Exception as e:
            print ('Error in Main Loop')
            print (e)
 


def mainprocess():

    try:
        print ('checking jobs list')
        gc.collect()

        #============================================================FACE IMAGE
        #get next FACE IMAGE job
        sql = "select jobid from jobs where job_completed is null and upper(jobtype) like 'FACE%';"
        face_image_joblist = pg_query(sql)

        for jobid in face_image_joblist:
            print (jobid[0])
            res=process_faceimg(jobid[0])

            sql = "select sender from jobs where jobid="+str(jobid[0])+';'
            replyto = pg_query(sql)

            if res:
                pdfkit.from_file("/workspace/webservices/job"+str(jobid[0])+"/results/matches.html", "/workspace/webservices/job"+str(jobid[0])+"/results/report"+str(jobid[0])+".pdf") 
                sendEmail( replyto[0][0], "/workspace/webservices/job"+str(jobid[0])+"/results/report"+str(jobid[0])+'.pdf')
            else:
                sendEmail( replyto[0][0], "/workspace/webservices/face_error.pdf")

            updatejob(jobid[0])

        #=======================================================YOLO 9000
        #get next YOLO IMAGE job
        sql = "select jobid from jobs where job_completed is null and upper(jobtype) like 'YOLO%';"
        yoloimg_joblist = pg_query(sql)
        
        for jobid in yoloimg_joblist:
            res=process_yolometa_img(jobid[0])

            if len(res) >0:

                sql = "select sender from jobs where jobid="+str(jobid[0])+';'
                replyto = pg_query(sql)

                fn="/workspace/webservices/job"+str(jobid[0])+"/results.txt"

                fout=open(fn,'w')
                fout.write('\n\n=====start======\n')

                for item in res:
                    fout.write('====='+item+'======\n')
                    
                    for cl in res[item]:
                        fout.write (cl+'   ')

                    fout.write('\n')

                fout.write('=====end======')
                fout.close()

                pdfkit.from_file("/workspace/webservices/job"+str(jobid[0])+"/results.txt", "/workspace/webservices/job"+str(jobid[0])+"/results.pdf") 
                sendEmail( replyto[0][0], "/workspace/webservices/job"+str(jobid[0])+"/results.pdf")

            else:
                print ('Job directory missing - JobID:' + str(jobid[0])) 

            updatejob(jobid[0])
        
        print ('all jobs completed for now')

            
    except Exception as e:
        print('Error in MAIN loop')
        print (e)
        gc.collect()



def restart():
    try:
        print('restart attempt...')
        time.sleep(30)
        main()
    except Exception as e:
        print(e)
        restart()


if __name__ == "__main__":
    main()

