# ROIAlign
ROIAlign from Mask-RCNN 

this is an implenment of ROIAlign from Mask-RCNN: https://arxiv.org/abs/1703.06870.
Considering there is no ROIAlign in matconvnet(http://www.vlfeat.org/matconvnet),
I just wrote this implenment so that we can use roialign layer in matlab.

the code is written in CUDA(.cu) and mex file which you can use it in matlab with GPU.
use the command in matlab:
```matlab
mexcuda roialign.cu
```
to compile the file and then you can use it.

you can add these codes to vl_simplenn.m in matconvnet

```matlab
case 'roialign'
  res(i+1).x = roialign(res(i).x,rois,l.subdivisions,l.transform);
```
this is forward propagation

```matlab
case 'roialign'
  res(i).dzdx = roialign(res(i).x,rois,res(i+1).dzdx,l.subdivisions,l.transform);
  ```
this is backward propagation
