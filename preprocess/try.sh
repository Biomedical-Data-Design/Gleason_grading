for i in `find ./train_image/ -type f -name "*tiff" `; 
do 
	base=$(basename $i); 
	name=${base/.tiff};
	mkdir -p ./patch/${name};
	echo -e "python preprocess_img.py --i "${base}" --annotation "${name}" --output "${name} >> run.sh; 
done

