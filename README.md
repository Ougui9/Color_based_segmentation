# Color based Segmentation

This is a program for object (target is a red barrel here) detection based on color segmentaion using Gaussian Mixture Model to simulate distribution of color points, and EM to compute probability of class a color point belonging to. In addition, to differentiate other objects with similar colors, I also add some shape conditions into it.

## Getting Started
1. Put all necessary to the folder specified in annotate.py and run annotate.py to annotate them in to several classes.
2. Change paras at the beginning train.py and start to train the model. (or you can skip 1,2 if you only want to try it.) 
2. put test image in he folder specified in test.py and run it. 

```
python test.py
```

### Prerequisites

Python 3


## test Results

![alt text](https://github.com/Ougui9/Color_based-_segmentation/blob/master/res/bbox001.png)
![alt text](https://github.com/Ougui9/Color_based-_segmentation/blob/master/res/bbox002.png)
![alt text](https://github.com/Ougui9/Color_based-_segmentation/blob/master/res/bbox003.png)
![alt text](https://github.com/Ougui9/Color_based-_segmentation/blob/master/res/bbox004.png)
![alt text](https://github.com/Ougui9/Color_based-_segmentation/blob/master/res/bbox005.png)
![alt text](https://github.com/Ougui9/Color_based-_segmentation/blob/master/res/bbox006.png)
![alt text](https://github.com/Ougui9/Color_based-_segmentation/blob/master/res/bbox007.png)
![alt text](https://github.com/Ougui9/Color_based-_segmentation/blob/master/res/bbox008.png)
![alt text](https://github.com/Ougui9/Color_based-_segmentation/blob/master/res/bbox009.png)
![alt text](https://github.com/Ougui9/Color_based-_segmentation/blob/master/res/bbox010.png)



## Reference
...https://en.wikipedia.org/wiki/Mixture_model
...https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm