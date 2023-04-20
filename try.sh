for i in `find /data/acharl15/gleason_grading/test_folder/ -name "Subset3*.tiff"`; 
do 
    base=$(basename $i); name=${base/.tiff}; 
    echo -e "python test_evaluate.py --i "$i" --annotation /data/acharl15/gleason_grading/test_folder/Subset2_Test_annotation/"$name"/ --output_patch_folder /data/acharl15/gleason_grading/test_folder/patch/"$name"/ --output_result_folder /data/acharl15/gleason_grading/test_folder/result_using_model_all/"$name"/ --model ./checkpoint/allSubset_epoch57.pth" >>Subsetall_test.sh; 
done
