# MAS & SAS
Research with [Dr.Sabuncu](https://scholar.google.com/citations?user=Pig-I4QAAAAJ&hl=en&oi=ao) and [A.Dalca](https://scholar.google.com/citations?user=zRy-zdAAAAAJ&hl=en&oi=ao)

Multi Atlas Segmentation of 3D Brain MRI Based on Unsupervised Learning 

## SAS(Single Atlas Segmentation)
![image](https://github.com/ShouYuqing/Images/blob/master/p1-1.png)
> Change the direction between volume data and atlas data while training: 
```python 
 train([atlas,volume],[volume,flow])
```
> Use different metrics.
> Use dice score to evaluate model.
### How does segmentation work
![image](https://github.com/ShouYuqing/Images/blob/master/p1-2.png)
## MAS(Multi Atlas Segmentation)
![image](https://github.com/ShouYuqing/Images/blob/master/p1-5.png)
>Label fusion 

>Models: MAS-2 MAS-3 MAS-4 MAS-5 (vm-1, vm-2 double)

>Spatial transform: linear/nearest

>Use dice score to evaluate model.

## Citation
Based on [Voxelmorph](https://arxiv.org/abs/1809.05231/) and [Unsupervised learning for registration](https://arxiv.org/abs/1805.04605v1/)


**Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration**  
[Adrian V. Dalca](http://adalca.mit.edu), [Guha Balakrishnan](http://people.csail.mit.edu/balakg/), [John Guttag](https://people.csail.mit.edu/guttag/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/)  
MICCAI 2018. [eprint arXiv:1805.04605](https://arxiv.org/abs/1805.04605)


**An Unsupervised Learning Model for Deformable Medical Image Registration**  
[Guha Balakrishnan](http://people.csail.mit.edu/balakg/), [Amy Zhao](http://people.csail.mit.edu/xamyzhao/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/), [John Guttag](https://people.csail.mit.edu/guttag/), [Adrian V. Dalca](http://adalca.mit.edu)  
CVPR 2018. [eprint arXiv:1802.02604](https://arxiv.org/abs/1802.02604)
