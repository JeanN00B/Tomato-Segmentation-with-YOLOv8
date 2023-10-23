# Tomato-Segmentation-with-YOLOv8
This is a repository with the results of my graduation work about detecting tomato maturity levels with YOLOv8, comparing with Mask R-CNN 

## Dataset Description:
The Laboro Tomato Dataset is a comprehensive dataset designed for object detection and instance segmentation. It features images of growing tomatoes in a greenhouse, categorized by their ripening stages and tomato types.
![dataset](https://github.com/JeanN00B/Tomato-Segmentation-with-YOLOv8/blob/main/Illustrations/tomatoes_categories.png)

### Key Features
- Ripening Stages: The dataset classifies tomatoes into three ripening stages: ripe, half-ripe, and green.
- Tomato Types: It includes two different types of tomatoes: cherry and regular.
- Annotations: For segmentation tasks, the dataset provides bounding box annotations and vertices representing tomato masks, along with class labels.

### Image Variabilirty
Images in this dataset are captured using two different cameras, resulting in variations in image quality and resolutions.

### Classification Criteria
- Tomato Types: Cherry tomatoes are considerably smaller than regular tomatoes.
- Ripeness States: Ripeness classification is based on the percentage of red color. Fully ripened tomatoes have 90% or more red color, half-ripened fall between 30-89%, and green tomatoes have 0-30% red color.
- Expert Validation: Other criteria were considered, and experts made the final classifications.

### Potential Applications
This dataset can be used for various applications, including:

- Harvest forecasts based on tomato maturity.
- Automatic harvest systems for ripe tomatoes.
- Targeted pesticide spraying on tomatoes at specific ripening stages.
- Quality control for yield production.

For detailed information, consult the original paper and examples in the dataset [LaboroTomato](https://github.com/laboroai/LaboroTomato).

## Setup the Environments
### YOLOv8 Virtual Environment

```bash
# Create or activate the YOLOv8 virtual environment

# Install ultralytics
pip install ultralytics

# Install Ray for hyperparameter tuning
pip install ray[tune]
```

### Mask R-CNN Virtual Environment
```bash
# Create or activate the Mask R-CNN virtual environment

# Install jedi (if not already installed)
pip install jedi>=0.10

# Install openmim
pip install openmim

# Install mmcv-full
mim install mmcv-full

# Install light-the-torch
pip install light-the-torch
ltt install torch torchvision

# Clone the mmdetection repository
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection

# Install mmdetection
pip install -e .
```


## Run the Codes
The Mask R-CNN and YOLOv8 jupyter notebooks, are designed to be run on google colab, just need to be aware to change the directories to the ones your project and dataset uses. The jupyter notebooks contains the code to train the Mask R-CNN and YOLOv8 models withouth hyperparameter tuning.

On the other hand, there are also some python codes, in the ![Color algorithm methodology](https://github.com/JeanN00B/Tomato-Segmentation-with-YOLOv8/tree/0ff5d5a80b465c7406b87f92b255b667aefa372a/Mask%20R-CNN%20code/Color%20algorithm%20methodology) there are the color analysis, and square error scripts used to simulate the color analysis techniques the benchmark paper authors applied. and in the ![Mask extractions and images](https://github.com/JeanN00B/Tomato-Segmentation-with-YOLOv8/tree/0ff5d5a80b465c7406b87f92b255b667aefa372a/Mask%20R-CNN%20code/Color%20algorithm%20methodology/Mask%20extractions%20and%20images) are the extracted masks and predicted images. Is important to considere that the groundtruth images used in the script are in the COCO format.

Finally, in the ![YOLOv8 code](https://github.com/JeanN00B/Tomato-Segmentation-with-YOLOv8/tree/0ff5d5a80b465c7406b87f92b255b667aefa372a/YOLOv8%20code) folder, there is the hyperparameter tuning script, this at the end will generate all the possible efficient hyperparameter combinations and trainings, also the best results obtained in this work are attached in the ![YOLOv8 model tuned results](https://github.com/JeanN00B/Tomato-Segmentation-with-YOLOv8/tree/0ff5d5a80b465c7406b87f92b255b667aefa372a/YOLOv8%20code/YOLOv8%20model%20tuned%20results)

