# delete the face encoding files - for load testing timings

# 453s on CPU 
# 7.5s on GPU
# for load testing the GPU delete the face encoding files

cd cjob1
rm -f search/*.fe_cnn
rm -f target/*.fe_cnn
rm -rf results
cd ..

./faceservice_main.py -i cjob1/ -j 123