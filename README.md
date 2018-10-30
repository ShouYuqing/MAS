# MAS & SAS
Multi Atlas Segmentation for 3D MRI brain image (in progress)

Based on Voxelmorph https://arxiv.org/abs/1809.05231
[Voxelmorph](https://arxiv.org/abs/1809.05231"")
## SAS(Single Atlas Segmentation)
> Change the direction between volume data and atlas data while training: 
```python 
 train([atlas,volume],[volume,flow])
```
> Use dice score to evaluate model, the code has been finished by 10/29/2018.
## MAS(Multi Atlas Segmentation)
