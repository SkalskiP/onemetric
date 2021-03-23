[![PyPI version](https://badge.fury.io/py/onemetric.svg)](https://badge.fury.io/py/onemetric)
![PyPI - License](https://img.shields.io/pypi/l/onemetric)
[![codecov](https://codecov.io/gh/SkalskiP/onemetric/branch/master/graph/badge.svg?token=ZFSEYF9WN4)](https://codecov.io/gh/SkalskiP/onemetric)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/onemetric)

<h1 align="center">onemetric</h1>

<p align="center"> 
    <img width="150" src="https://onemetric-images.s3.eu-central-1.amazonaws.com/favicon.png" alt="Logo">
</p>

## Installation

* Install onemetric from PyPI (recommended):

   ```console
   pip install onemetric
   ```
  
* Install onemetric from the GitHub source:

   ```console
   git clone https://github.com/SkalskiP/onemetric.git
   cd onemetric
   python setup.py install
   ```

## Example

<p align="center"> 
    <img width="800" src="https://onemetric-images.s3.eu-central-1.amazonaws.com/sample.png" alt="dataset-sample">
</p>

**Figure 1.** Dataset sample, blue - ground-truth and red - detection.

### Calculate mAP@0.5

```python
>>> from onemetric.cv.loaders import YOLOLoader
>>> from onemetric.cv.object_detection import MeanAveragePrecision

>>> model = load_model(...)  # model-specific loading method

>>> data_set = YOLOLoader(
...     images_dir_path=DATA_SET_IMAGES_PATH, 
...     annotations_dir_path=DATA_SET_ANNOTATIONS_PATH
... ).load()

>>> true_batches, detection_batches = [], []
>>> for entry in data_set:
>>>     detections = model(entry.get_image())  # model-specific prediction method
>>>     true_batches.append(entry.get_annotations())
>>>     detection_batches.append(detections)

>>> mean_average_precision = MeanAveragePrecision.from_detections(
...     true_batches=true_batches, 
...     detection_batches=detection_batches, 
...     num_classes=12,
...     iou_threshold=0.5
... )

>>> mean_average_precision.value
0.61
```

### Calculate Confusion Matrix

```python


>>> confusion_matrix = ConfusionMatrix.from_detections(
...     true_batches=true_batches, 
...     detection_batches=detection_batches,
...     num_classes=12
... )

>>> confusion_matrix.plot(CONFUSION_MATRIX_TARGET_PATH, class_names=CLASS_NAMES)
```

<p align="center"> 
    <img width="800" src="https://onemetric-images.s3.eu-central-1.amazonaws.com/confusion_matrix.png" alt="dataset-sample">
</p>

**Figure 2.** Create confusion matrix chart

## Documentation

The official documentation is hosted on Github Pages: https://skalskip.github.io/onemetric

## Contribute

Feel free to file [issues](https://github.com/SkalskiP/onemetric/issues) or [pull requests](https://github.com/SkalskiP/onemetric/pulls). **Let us know what metrics should be part of onemetric!**

## Citation

Please cite onemetric in your publications if this is useful for your research. Here is an example BibTeX entry:

```BibTeX
@MISC{onemetric,
   author = {Piotr Skalski},
   title = {{onemetric}},
   howpublished = "\url{https://github.com/SkalskiP/onemetric/}",
   year = {2021},
}
```

## License

This project is licensed under the BSD 3 - see the [LICENSE][1] file for details.


[1]: https://github.com/SkalskiP/onemetric/blob/master/LICENSE
