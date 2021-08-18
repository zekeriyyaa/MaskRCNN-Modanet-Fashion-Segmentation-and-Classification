# MaskRCNN-Modanet-Fashion-Segmentation-and-Classification
With the contributions of [Sergen](https://github.com/SergenAsik) and [Murat](https://github.com/MuratYavuzz).

The purpose of this project is classification of clothes under season. There are three main subprocess exist as given below:
1. Training Model-1 from Modanet Dataset and Generate Custom Dataset's Labels
2. Training Model-2 from Custom Dataset
3. Evaluate test results

![](https://github.com/zekeriyyaa/MaskRCNN-Modanet-Fashion-Segmentation-and-Classification/blob/main/SystemArchitecture.PNG)

For training the model, it is needed to labeled images but our custom dataset is not labeled. There are so many image to label and **we do not want to waste time for it**. The our idea which can handle this issue is use prelabeled fashion dataset to train a model. And then, our custom dataset is given as input into produced model and got segmentation of clothes as result. From this segmentation result's mask, the labels are produces for each clothes in each images.

In this way, the labels are produced by automatically from using model1. After that, our custom dataset's labels is ready to train for classification of season.

## 1. Training Model1 from Modanet Dataset and Generate Custom Dataset's Labels
### Prepare Modanet Dataset
[Modanet](https://github.com/eBay/modanet)  dataset has over 40GB fashion clothes images and its annotations. While downloading, it is possible to choice which 40GB or 2GB dataset. In order to more manageable, we prefer 2GB dataset. The annotation data format of ModaNet follows the same style as [COCO-dataset](http://cocodataset.org).

We suggest that use google colab to download dataset. There is our [Modanet-Download-Colab-File](https://github.com/zekeriyyaa/MaskRCNN-Modanet-Fashion-Segmentation-and-Classification/blob/main/modaNetDatasetDownload.ipynb).

After download, there are over 10.000 images and json file which is include their labels. We parsed dataset into %80 [instances_train.json](https://github.com/zekeriyyaa/MaskRCNN-Modanet-Fashion-Segmentation-and-Classification/blob/main/modanetAnnotations/instances_train.json), %10 [instances_test.json](https://github.com/zekeriyyaa/MaskRCNN-Modanet-Fashion-Segmentation-and-Classification/blob/main/modanetAnnotations/instances_test.json) and %10 [instances_val.json](https://github.com/zekeriyyaa/MaskRCNN-Modanet-Fashion-Segmentation-and-Classification/blob/main/modanetAnnotations/instances_val.json). It is remember that the json file is also need to seperate as given percentages. 

But there is an issue that about annotations type. The modanet is provide us annotations with coco format but we used [Vgg-annotation](https://roboflow.com/formats/via-json) type. Therefore, we converted parsed coco json files to vgg json files and located them into [train](https://github.com/zekeriyyaa/MaskRCNN-Modanet-Fashion-Segmentation-and-Classification/tree/main/model1/samples/custom/dataset/train) and [validation](https://github.com/zekeriyyaa/MaskRCNN-Modanet-Fashion-Segmentation-and-Classification/tree/main/model1/samples/custom/dataset/val) folder as namely **via_region_data.json**.

As a result, we have given annotations format:
```
{
 "377": {
  "filename": "0000377.jpg",
  "size": -1,
  "regions": [
   {
    "shape_attributes": {
     "name": "polygon",
     "all_points_x": [
      244,
      247,
      248 ...
     ],
     "all_points_y": [
      234,
      257,
      271 ...
     ]
    },
    "region_attributes": {
     "fashion": "1"
    }
   }
  ],
  "file_attributes": {
   "width": 400,
   "height": 600
  }
 }
}
```

The dataset should be located into [dataset](https://github.com/zekeriyyaa/MaskRCNN-Modanet-Fashion-Segmentation-and-Classification/tree/main/model1/samples/custom/dataset) folder with its annotations json file seperately as [train](https://github.com/zekeriyyaa/MaskRCNN-Modanet-Fashion-Segmentation-and-Classification/tree/main/model1/samples/custom/dataset/train) and [validation](https://github.com/zekeriyyaa/MaskRCNN-Modanet-Fashion-Segmentation-and-Classification/tree/main/model1/samples/custom/dataset/val). The [test images](https://github.com/zekeriyyaa/MaskRCNN-Modanet-Fashion-Segmentation-and-Classification/tree/main/model1/images) are located into images folder which is located into base folder. 

### Labels
Each polygon (bounding box, segmentation mask) annotation is assigned to one of the following labels:

| Label | Description | Fine-Grained-categories |
| --- | --- | --- |
| 1 | bag | bag |
| 2 | belt | belt |
| 3 | boots | boots |
| 4 | footwear | footwear |
| 5 | outer | coat/jacket/suit/blazers/cardigan/sweater/Jumpsuits/Rompers/vest |
| 6 | dress | dress/t-shirt dress |
| 7 | sunglasses | sunglasses |
| 8 | pants | pants/jeans/leggings |
| 9 | top | top/blouse/t-shirt/shirt |
|10 | shorts | shorts |
|11 | skirt | skirt |
|12 | headwear | headwear |
|13 | scarf & tie | scartf & tie |

### Train Model-1 using Modanet Dataset
Our custom [training](https://github.com/zekeriyyaa/MaskRCNN-Modanet-Fashion-Segmentation-and-Classification/blob/main/model1/samples/custom/trainModaNet.py) and [test](https://github.com/zekeriyyaa/MaskRCNN-Modanet-Fashion-Segmentation-and-Classification/blob/main/model1/samples/custom/testModaNet.py) files is located into given folder as link. You can start to training without any pretrained model as give model parameter as ```--model=coco``` . In this way, the coco weighted model will downloaded automatically. In other option is use own model.

[![](https://img.shields.io/badge/keras-2.3.1-blue)](https://github.com/keras-team/keras/releases)
[![](https://img.shields.io/badge/tensorflow-1.15.2-blue)](https://www.tensorflow.org/install/pip)


You can run command directly from the command line as such:
```
# Train a new model starting from pre-trained COCO weights
python3 model1/samples/custom/trainModaNet.py train --dataset=/path/to/coco/ --model=coco

# Continue training a model that you had trained earlier
python3 model1/samples/custom/trainModaNet.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

# For our model
python3 model1/samples/custom/trainModaNet.py train --dataset=/path/to/coco/ --model=mask_rcnn_Modanet.h5
```

You can also run the evaluation code with:
```
# Run evaluation code on the last trained model
python3 testModaNet.py
```

You can access our trained model [mask_rcnn_Modanet.h5](https://drive.google.com/file/d/1XEg4wqdz1G4yTcjeakBUh22YDw8Jf3f4/view?usp=sharing) (~244MB) and must be located it [model1](https://github.com/zekeriyyaa/MaskRCNN-Modanet-Fashion-Segmentation-and-Classification/tree/main/model1) as mask_rcnn_Modanet.h5 .

### Segmentation Results
Sample segmentation result is shown as below: 
![](https://github.com/zekeriyyaa/MaskRCNN-Modanet-Fashion-Segmentation-and-Classification/blob/main/segmentationResult.PNG)

### Generate Custom Dataset's Labels
Our custom dataset was passing into trained model **mask_rcnn_Modanet.h5** as input. As a result, we got segmentation of clothes. The segmentation result has mask of segmented clothes and their class ID. Using this mask we produced detected clothes's boundaries and their labels as vgg annotation type. This is how we can handle the issue which is already talked above as our unlabel dataset.

The [generateVggAnno.py](https://github.com/zekeriyyaa/MaskRCNN-Modanet-Fashion-Segmentation-and-Classification/blob/main/model1/samples/custom/generateVggAnno.py) file is used for generate vgg annotations from our custom dataset. It is used **mask_rcnn_Modanet.h5** model for images segmentation and labelling them.
```
# Run evaluation code on the last trained model
python3 generateVggAnno.py
```
After the run **generateVggAnno.py**, the **custom_dataset_annotations.json** is produced as a result on the same folder. We use this annotations file for training model-2. 

## 2. Training Model-2 from Custom Dataset
### Prepare Custom Dataset
After the annotations are generated from previous step as **custom_dataset_annotations.json**, the json file and also dataset is parsed into %80 [train](https://github.com/zekeriyyaa/MaskRCNN-Modanet-Fashion-Segmentation-and-Classification/tree/main/model2/samples/custom/dataset/train), %10 [validation](https://github.com/zekeriyyaa/MaskRCNN-Modanet-Fashion-Segmentation-and-Classification/tree/main/model2/samples/custom/dataset/val) and test as given percentage. The annotations files for train and validation images namely **via_region_data.json** are also located into same folder with train and validation images under dataset folder. The [test images](https://github.com/zekeriyyaa/MaskRCNN-Modanet-Fashion-Segmentation-and-Classification/tree/main/model2/images) are located into images folder which is located into base folder.

### Labels
Our custom dataset has normally 10 different fashion and its season labels. We can get class name and season label from images file name. In the **generateVggAnno.py**, these file names are used for generate annotations. Actually each class has 5 subclass which specified seasons as summer, winter, autumn, spring and all. Because of unsufficient number of sample, we had to decrease class label and just use the most exist ones.

Each polygon (bounding box, segmentation mask) annotation is assigned to one of the following labels: 

usage: fashionClassName_category

Categories: yaz->summer, sonb->autumn, ilkb->spring, kis->winter, 4mev->all

| Label | Description |
| --- | --- |
| 1 | skirt_yaz |
| 2 | bag_yaz |
| 3 | footwear_sonb |
| 4 | top_yaz |
| 5 | bag_4mev |
| 6 | footwear_yaz |
| 7 | footwear_kis |
| 8 | shorts_yaz |
| 9 | skirt_ilkb |

### Train Model-2 using Custom Dataset
Our custom [training](https://github.com/zekeriyyaa/MaskRCNN-Modanet-Fashion-Segmentation-and-Classification/blob/main/model2/samples/custom/trainSeasonData.py) and [test](https://github.com/zekeriyyaa/MaskRCNN-Modanet-Fashion-Segmentation-and-Classification/blob/main/model1/samples/custom/testSeason.py) files is located into given folder as link. You can start to training without any pretrained model as give model parameter as ```--model=coco``` . In this way, the coco weighted model will downloaded automatically. In other option is use own model. But we prefer use our pretrained model produced from model-1 ```--model=mask_rcnn_Modanet.h5-rcnn-```

You can run command directly from the command line as such:
```
# Train a new model starting from pre-trained COCO weights
python3 model2/samples/custom/trainSeasonData.py train --dataset=/path/to/coco/ --model=coco

# Continue training a model that you had trained earlier
python3 model2/samples/custom/trainSeasonData.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

# For our model
python3 model2/samples/custom/trainSeasonData.py train --dataset=/path/to/coco/ --model=mask_rcnn_Modanet.h5
```

You can access our trained model [mask_rcnn_Season.h5](https://drive.google.com/file/d/1V-SOJiEyIdidsY_1r8SQ8JS-vQqVH5Fi/view?usp=sharing) (~244MB) and must be located it [model2](https://github.com/zekeriyyaa/MaskRCNN-Modanet-Fashion-Segmentation-and-Classification/tree/main/model2) as mask_rcnn_Season.h5 .

## 3. Evaluate test results
After the training of model-2 from custom dataset, the **mask_rcnn_Modanet.h5** is produced as result. And now using this model, we can get seasons of clothes from given images as input.

You can run the evaluation code with:
```
# Run evaluation code on the last trained model
python3 testSeason.py
```

### Compare Model-1 and Model-2 Results
Model-1 Outputs            |  Model-2 Outputs
:-------------------------:|:-------------------------:
![](https://github.com/zekeriyyaa/MaskRCNN-Modanet-Fashion-Segmentation-and-Classification/blob/main/outputs/bag_4mev_112_model1.png)  |  ![](https://github.com/zekeriyyaa/MaskRCNN-Modanet-Fashion-Segmentation-and-Classification/blob/main/outputs/bag_4mev_112_model2.png)
![](https://github.com/zekeriyyaa/MaskRCNN-Modanet-Fashion-Segmentation-and-Classification/blob/main/outputs/bag_4mev_129_model1.png) | ![](https://github.com/zekeriyyaa/MaskRCNN-Modanet-Fashion-Segmentation-and-Classification/blob/main/outputs/bag_4mev_129_model2.png)







