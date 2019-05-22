### SSD Traffic Lights Detection



*update*:



Now we have a more advanced implementation for traffic light detection, the source codes can be found at **strangeai.pro**: [link](http://manaai.cn/aicodes_detail3.html?id=26)



it's more accurate, more faster, and more robustic in traffic light detection, also need fewer memory usage.



### Prepare Data

This step is the same as my previous repo kitti_ssd, train kitti dataset on ssd, this repo using dataset is LISA, a traffic light detection dataset. As the same of previous, we get `Images` and `Label` folder which contains all images and labels, one image reflect on **same named** label file, in `./data` folder we have a script `gen_all_labels.sh` to generate same named label files contains nothing, just in case some images has no label bound box, this may cause unable to train SSD.

### Generate lmdb data
To train caffe, we'd better use lmdb database to feed data into network, since we got images and labels, oh, another thing I just forgot, **label file format as follow**:
```
class_index x_min y_min x_max y_max
```
Continue, to generate lmdb data you only need to type:
```
bash ./data/create_list.sh
```
This will generate `trainval.txt` , `test.txt` and `test_name_size.txt`. In `create_list.sh` you have to change your dataset name, and **we suppose your data placed in ~/data**. So, if you got a dataset named FACE, then your trainval.txt and test.txt will seems like this:
```
Images/0001.jpg Labels/0001.txt
```
Simple, right? and then:
```
bash ./data/create_data.sh
```
In the sh file, we have a data_dir viraible, **please make sure data_dir + trainval.txt line** can reach your real image, otherwise you cannot generate lmdb file, or just get zero lenght.
**After this step, you are going have trainval.lmdb and test.lmdb** under `~/data/YOURDATASET/lmdb`

### Predict Using SSD
ok, before train SSD, you can test this, you can download 2 model, [VGG16.v2.caffemodel](http://pan.baidu.com/s/1geVTWen) and [VGG_LISA_SSD_414x125_iter_120000.caffemodel](http://pan.baidu.com/s/1boPq4I7)`, the first one is SSD pretrain model, you gonna need it when you train SSD, and second is I trained model using for traffic light detection.
**Place first model into ./models/VGGNet/LISA/SSD_414x125**
**Place second model into ./models/VGGNet**
And then type:
```
python ssd_detection.py
```
You will see something like this:

![Picture](http://ojek5ksya.bkt.clouddn.com/NTAk0jDb5gxUfmrS.png)

### Train SSD
To train SSD, you just need:
```
python train_ssd_lisa.py
```
But you need to change some directory and datset names.

### Predict Many Images
We provide a script to predict all images under an dir, using `ssd_detection_all.py` instead.

### Copyright
(c) DiDiChuXing All Rights Reserved
