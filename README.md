# MAS & SAS
Multi Atlas Segmentation for 3D MRI brain image 

Based on [Voxelmorph](https://arxiv.org/abs/1809.05231/) and [Unsupervised learning for registration](https://arxiv.org/abs/1805.04605v1/)
## SAS(Single Atlas Segmentation)
> Change the direction between volume data and atlas data while training: 
```python 
 train([atlas,volume],[volume,flow])
```
> Use dice score to evaluate model, code been finished by 10/29/2018.
## MAS(Multi Atlas Segmentation)
>Label fusion is implemented in the test part of MAS

>Models: MAS-2 MAS-3 MAS-4 MAS-5

>Use dice score to evaluate model, code been finished by 11/5/2018

## Citation
**Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration**  
[Adrian V. Dalca](http://adalca.mit.edu), [Guha Balakrishnan](http://people.csail.mit.edu/balakg/), [John Guttag](https://people.csail.mit.edu/guttag/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/)  
MICCAI 2018. [eprint arXiv:1805.04605](https://arxiv.org/abs/1805.04605)


**An Unsupervised Learning Model for Deformable Medical Image Registration**  
[Guha Balakrishnan](http://people.csail.mit.edu/balakg/), [Amy Zhao](http://people.csail.mit.edu/xamyzhao/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/), [John Guttag](https://people.csail.mit.edu/guttag/), [Adrian V. Dalca](http://adalca.mit.edu)  
CVPR 2018. [eprint arXiv:1802.02604](https://arxiv.org/abs/1802.02604)
