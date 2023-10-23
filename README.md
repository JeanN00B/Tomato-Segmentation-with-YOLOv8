# Tomato-Segmentation-with-YOLOv8
This is a repository with the results of my graduation work about detecting tomato maturity levels with YOLOv8, comparing with Mask R-CNN 

## Dataset Description:
The Laboro Tomato Dataset is a comprehensive dataset designed for object detection and instance segmentation. It features images of growing tomatoes in a greenhouse, categorized by their ripening stages and tomato types.![dataset](https://github.com/JeanN00B/Tomato-Segmentation-with-YOLOv8/blob/main/Illustrations/tomatoes_categories.png)

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
