#!/bin/bash
python test_evaluate.py --i /data/acharl15/gleason_grading/test_folder/Subset3_Test_9_Philips.tiff --annotation /data/acharl15/gleason_grading/test_folder/Subset3_Test_annotation/Philips/Subset3_Test_9_Philips/ --output_patch_folder /data/acharl15/gleason_grading/test_folder/patch/Subset3_Test_9_Philips/ --output_result_folder /data/acharl15/gleason_grading/test_folder/result_using_model_all/Subset3_Test_9_Philips/ --model ./checkpoint/allSubset_epoch57.pth
python test_evaluate.py --i /data/acharl15/gleason_grading/test_folder/Subset3_Test_6_Zeiss.tiff --annotation /data/acharl15/gleason_grading/test_folder/Subset3_Test_annotation/Zeiss/Subset3_Test_6_Zeiss/ --output_patch_folder /data/acharl15/gleason_grading/test_folder/patch/Subset3_Test_6_Zeiss/ --output_result_folder /data/acharl15/gleason_grading/test_folder/result_using_model_all/Subset3_Test_6_Zeiss/ --model ./checkpoint/allSubset_epoch57.pth
python test_evaluate.py --i /data/acharl15/gleason_grading/test_folder/Subset3_Test_9_Zeiss.tiff --annotation /data/acharl15/gleason_grading/test_folder/Subset3_Test_annotation/Zeiss/Subset3_Test_9_Zeiss/ --output_patch_folder /data/acharl15/gleason_grading/test_folder/patch/Subset3_Test_9_Zeiss/ --output_result_folder /data/acharl15/gleason_grading/test_folder/result_using_model_all/Subset3_Test_9_Zeiss/ --model ./checkpoint/allSubset_epoch57.pth
python test_evaluate.py --i /data/acharl15/gleason_grading/test_folder/Subset3_Test_4_KFBio.tiff --annotation /data/acharl15/gleason_grading/test_folder/Subset3_Test_annotation/KFBio/Subset3_Test_4_KFBio/ --output_patch_folder /data/acharl15/gleason_grading/test_folder/patch/Subset3_Test_4_KFBio/ --output_result_folder /data/acharl15/gleason_grading/test_folder/result_using_model_all/Subset3_Test_4_KFBio/ --model ./checkpoint/allSubset_epoch57.pth
python test_evaluate.py --i /data/acharl15/gleason_grading/test_folder/Subset3_Test_11_Akoya.tiff --annotation /data/acharl15/gleason_grading/test_folder/Subset3_Test_annotation/Akoya/Subset3_Test_11_Akoya/ --output_patch_folder /data/acharl15/gleason_grading/test_folder/patch/Subset3_Test_11_Akoya/ --output_result_folder /data/acharl15/gleason_grading/test_folder/result_using_model_all/Subset3_Test_11_Akoya/ --model ./checkpoint/allSubset_epoch57.pth
python test_evaluate.py --i /data/acharl15/gleason_grading/test_folder/Subset3_Test_3_Zeiss.tiff --annotation /data/acharl15/gleason_grading/test_folder/Subset3_Test_annotation/Zeiss/Subset3_Test_3_Zeiss/ --output_patch_folder /data/acharl15/gleason_grading/test_folder/patch/Subset3_Test_3_Zeiss/ --output_result_folder /data/acharl15/gleason_grading/test_folder/result_using_model_all/Subset3_Test_3_Zeiss/ --model ./checkpoint/allSubset_epoch57.pth
python test_evaluate.py --i /data/acharl15/gleason_grading/test_folder/Subset3_Test_4_Philips.tiff --annotation /data/acharl15/gleason_grading/test_folder/Subset3_Test_annotation/Philips/Subset3_Test_4_Philips/ --output_patch_folder /data/acharl15/gleason_grading/test_folder/patch/Subset3_Test_4_Philips/ --output_result_folder /data/acharl15/gleason_grading/test_folder/result_using_model_all/Subset3_Test_4_Philips/ --model ./checkpoint/allSubset_epoch57.pth
python test_evaluate.py --i /data/acharl15/gleason_grading/test_folder/Subset3_Test_9_Akoya.tiff --annotation /data/acharl15/gleason_grading/test_folder/Subset3_Test_annotation/Akoya/Subset3_Test_9_Akoya/ --output_patch_folder /data/acharl15/gleason_grading/test_folder/patch/Subset3_Test_9_Akoya/ --output_result_folder /data/acharl15/gleason_grading/test_folder/result_using_model_all/Subset3_Test_9_Akoya/ --model ./checkpoint/allSubset_epoch57.pth
python test_evaluate.py --i /data/acharl15/gleason_grading/test_folder/Subset3_Test_1_Zeiss.tiff --annotation /data/acharl15/gleason_grading/test_folder/Subset3_Test_annotation/Zeiss/Subset3_Test_1_Zeiss/ --output_patch_folder /data/acharl15/gleason_grading/test_folder/patch/Subset3_Test_1_Zeiss/ --output_result_folder /data/acharl15/gleason_grading/test_folder/result_using_model_all/Subset3_Test_1_Zeiss/ --model ./checkpoint/allSubset_epoch57.pth