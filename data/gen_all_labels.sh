# !/bin/bash

# map all images into labels with same filename 00001.jpg -> 00001.txt
current_dir=`pwd`
des_dir=${current_dir}"/labels"

image_dir="/dayTraining"
image_list="imglist"
ls ${current_dir}${image_dir}|grep ".png" > ${image_list}

echo $des_dir
if [ ! -d "${des_dir}" ]; then
	mkdir ${des_dir}
fi

for line in `cat ${image_list}`
do
	file_name=${line%.*}
	touch ${des_dir}"/"${file_name}".txt"
done

rm ${image_list}
echo "done!"