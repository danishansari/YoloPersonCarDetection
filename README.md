# YoloPersonCarDetection
Assignment on detection of person and car on a specific dataset.
Using yolov5 to perform the task.

### Dependencies:
```pip install -r yolov5/requirements.txt```

### Trained Model:
https://drive.google.com/file/d/1RZiSYfdzGhIBMvA08KO7qqpGzdgSTTzg/view?usp=sharing

### Data Visualization:
```python scripts/visualize.py trainval/annotations/bbox-annotations.json```

### Data Conversion:
Convert data from bdbox to yolo-format
```python scripts/convert2yolo.py </path/to/annotations/json/file>```

### Data Augmentation:
``` python scripts/scale_augment.py trainval/images/train/```

### Evaluation:
```python val.py --weights best.pt --data ../tainval/person_car.yaml --img 640```

### Inference:
```python detect.py --weights </path/to/model/> --souce ../trainval/images --view-img```
