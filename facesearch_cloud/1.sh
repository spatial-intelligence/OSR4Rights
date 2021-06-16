# delete the face encoding files - for load testing timings
# for load testing the GPU delete the face encoding files
rm -rf cjob1/search/*.fe_cnn
rm -rf cjob1/target/*.fe_cnn

./faceservice_main.py -i cjob1/ -j 123