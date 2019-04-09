# DeformConvnet-pytorch-chapter4

The chapter4 of the segmentation network summary: 
### Pay attention to the internal structure of CNN.

External links: Deformable Convolutional Networks [paper](https://arxiv.org/abs/1703.06211).

Nowadays, a lot of CNN networks perform very well in image segmentation and classification. However, we should not only focus on how to build a CNN network architecture efficiently, but also pay attention to the implementation of the internal structure of CNN.

As far as I am concerned, the idea of this paper is particularly excellent and novel. Starting from two most fundamental operations of CNN, it rethinks and improves the operation of convolution and pooling. And the main idea of this paper is to put forward the "offset" idea.

The illustration in the paper:

![image](https://github.com/hydxqing/DeformConvnet-pytorch-chapter4/blob/master/picture_in_paper/deformable_convolution.png)
![image](https://github.com/hydxqing/DeformConvnet-pytorch-chapter4/blob/master/picture_in_paper/deformable_RoI_pooling.png)

##### We modified the code and embedded the deformable convolutional layer into the EDANet(We'll talk about it later) model, successfully trained and tested our own data set.

***References***

This code borrows from [Wei Ouyang](https://github.com/oeway)'s [work](https://github.com/oeway/pytorch-deform-conv) and is modified to use on my own dataset.

### Environment: 
  
            Pytorch version >> 0.4.1; [Python 2.7]
            
## Notes
1. Someone once said something that I think is great: The weight method is replaced by the bilinear method, that is, the sampling method instead of the weight method. This thinking can be spread out to do more work.
2. Some blogs: https://www.cnblogs.com/daihengchen/p/6880774.html
