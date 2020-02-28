## facemask

------------
### Introduction

Face_mask is a fast mask detection classifier that utilizes pytorch

------------

### Installation

Try and start with

```
pip install facemask
```

-------------

### Use

download the pretrained model from [BaiDu](https://pan.baidu.com/s/111neutXya4cwYpqUUYQ7ew), password: fq1s  

```python
import facemask
model = facemask.FaceMaskDetector(modelpath)

model.detect_image_show(image_path)
# or just return the detect results
bboxs = model.model(image_path)             # bbox_coordinate, confidence, label
```