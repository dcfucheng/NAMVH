# NAMVH
This repository provides the implementation of *NAMVH* as described in the papers:

> Nonlinear Asymmetric Multi-Valued Hashing<br>
> Cheng Da, Gaofeng Meng, Shiming Xiang, Kun Ding, Shibiao Xu, Qing Yang, and Chunhong Pan<br>
> TPAMI, 2018.<br>


Meanwhile, this repository also provides the implementation of *AMVH* as described in the papers:

> AMVH: Asymmetric Multi-Valued Hashing<br>
> Cheng Da, Shibiao Xu, Kun Ding, Gaofeng Meng, Shiming Xiang, Chunhong Pan<br>
> CVPR, 2017.<br>

# Prerequisites

Linux 16.04.5
NVIDIA GPU + CUDA-8.0 and corresponding CuDNN
Caffe
Matlab 2015b

# Installation Guide

You need to compile the modified Caffe library in this repository. You can consult the generic [Caffe installation guide](http://caffe.berkeleyvision.org/installation.html).

When the dependencies of Caffe are installed, try:

```
make
```

If there are no error messages, then you can compile and install the matlab wrappers:
```
make matcaffe
```

# Usage 
run AMVH for AMVH: Asymmetric Multi-Valued Hashing

```
cd NAMVH
matlab -r -nodesktop run_AMVH
```

#### results
    maps =

    0.4362    0.4906
    0.4953    0.5190


run NAMVH for Nonlinear Asymmetric Multi-Valued Hashing

```
cd NAMVH
matlab -r -nodesktop run_AMVH
```

#### results
    maps =

    0.4822    0.5045
    0.5208    0.5599


# Datasets
We provide ESP-GAME dataset to test our codes. [./data](https://github.com/dcfucheng/NAMVH/tree/master/data),

### Citing
If you find *NAMVH* and *AMVH* useful in your research, we ask that you cite the following paper:

	@ARTICLE{8451950, 
	author={C. Da and G. Meng and S. Xiang and K. Ding and S. Xu and Q. Yang and C. Pan}, 
	journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
	title={Nonlinear Asymmetric Multi-Valued Hashing}, 
	year={2018}, 
	volume={}, 
	number={}, 
	pages={1-1}, 
	keywords={Databases;Binary codes;Optimization;Neural networks;Encoding;Hamming distance;Semantics;Asymmetric hashing;multi-valued embeddings;binary sparse representation;nonlinear transformation}, 
	doi={10.1109/TPAMI.2018.2867866}, 
	ISSN={0162-8828}, 
	month={},}
	
	@inproceedings{DBLP:conf/cvpr/DaXDMXP17,
  author    = {Cheng Da and
               Shibiao Xu and
               Kun Ding and
               Gaofeng Meng and
               Shiming Xiang and
               Chunhong Pan},
  title     = {{AMVH:} Asymmetric Multi-Valued hashing},
  booktitle = {2017 {IEEE} Conference on Computer Vision and Pattern Recognition,
               {CVPR} 2017, Honolulu, HI, USA, July 21-26, 2017},
  pages     = {898--906},
  year      = {2017},
  crossref  = {DBLP:conf/cvpr/2017},
  url       = {https://doi.org/10.1109/CVPR.2017.102},
  doi       = {10.1109/CVPR.2017.102},
  timestamp = {Tue, 14 Nov 2017 17:15:06 +0100},
  biburl    = {https://dblp.org/rec/bib/conf/cvpr/DaXDMXP17},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
	
	
	