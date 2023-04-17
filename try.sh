for i in `find /data/acharl15/gleason_grading/test_folder/ -name "Subset1*.tiff"`; 
do 
    base=$(basename $i); name=${base/.tiff}; 
    echo -e "python test_evaluate.py --i "$i" --annotation /data/acharl15/gleason_grading/test_folder/Subset1_Test_annotation/"$name"/ --output_patch_folder /data/acharl15/gleason_grading/test_folder/patch/"$name"/ --output_result_folder /data/acharl15/gleason_grading/test_folder/result/"$name"/ --model ./checkpoint/Subset1_epoch36.pth" >>Subset1_test.sh; 
done
