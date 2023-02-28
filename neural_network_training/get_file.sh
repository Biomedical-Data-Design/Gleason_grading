for i in `find ../data-acharl15/gleason_grading/ -name "Subset2*.tiff"`; do base=$(basename $i); name=${base/.tiff}; echo -e ${name}","${base}","${name} >>file.txt; done
