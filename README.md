## Exploring Balanced Feature Spaces for Representation Learning 

<p align="center">
  <img src="https://user-images.githubusercontent.com/17242808/106268737-75f35680-6266-11eb-8a79-c15dbf2c03b5.jpg" width="600">
</p>


This is a PyTorch implementation of the paper:

**Exploring Balanced Feature Spaces for Representation Learning**  
[Bingyi Kang](https://scholar.google.com.sg/citations?user=NmHgX-wAAAAJ&hl=en), [Yu li], [Zehuan Yuan](https://shallowyuan.github.io/), [Jiashi Feng](https://sites.google.com/site/jshfeng/)  
[[OpenReview](https://openreview.net/forum?id=OqtLIabPTit)]
[[Arxiv]()]
[[PDF]()] 
_International Conference on Learning Representations (ICLR), 202_

```
@inproceedings{kang2021exploring,
  title={Exploring Balanced Feature Spaces for Representation Learning},
  author={Bingyi Kang and Yu Li and Sa Xie and Zehuan Yuan and Jiashi Feng},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=OqtLIabPTit}
}



### Preparation

This code is based on the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet), the [official MoCo code](https://github.com/facebookresearch/moco) and the [official decoupling code for long-tailed recognition](https://github.com/facebookresearch/classifier-balancing).

Install PyTorch and ImageNet accordingly. 



### Datasets

We created six datasets from ImageNet that have a similar number of labeled images to [ImageNet-LT](https://arxiv.org/abs/1904.05160) but with varying distributions as shown by the following figure. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/17242808/106269437-77714e80-6267-11eb-97f7-72328ed968b5.png" width="400">
</p>

The statistics of these datasets are given in the following table.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">ImageNet-</th>
<th valign="bottom">Max</th>
<th valign="bottom">Min</th>
<th valign="bottom">Total</th>
<!-- TABLE BODY -->
<tr><td align="left">LT0</td>
<td align="center">115</td>
<td align="center">115</td>
<td align="center">115,000</td>
</tr>
<tr><td align="left">LT2</td>
<td align="center">204</td>
<td align="center">67</td>
<td align="center">115,885</td>
</tr>
</tr>
<tr><td align="left">LT4</td>
<td align="center">343</td>
<td align="center">37</td>
<td align="center">115,801</td>
</tr>
</tr>
<tr><td align="left">LT6</td>
<td align="center">553</td>
<td align="center">20</td>
<td align="center">115,836</td>
</tr>
</tr>
<tr><td align="left">LT8</td>
<td align="center">857</td>
<td align="center">10</td>
<td align="center">115,852</td>
</tr>
</tr>
<tr><td align="left">LT</td>
<td align="center">1280</td>
<td align="center">5</td>
<td align="center">115,846</td>
</tr>
</tbody></table>


### Loss study 

#### Contrastive Loss (MoCo V1)

Note: The following command line use ImageNet as an example, 

- Representation Learning 
```
python3 main_moco.py \
  -a resnet50 \
  --lr 0.03 \
  --weight-decay 0.0005 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --subset data/imagenet_longtail_train.txt  \
  --name longtail_moco_resnet50 \
  path/to/imagenet
```

- Classifier Learning 
```
python3 main_lincls.py -a resnet50 --dist --pretrained longtail_moco_resnet50.pth.tar \
--name longtail2full_moco_resnet50
```

- Supervised Cross-Entropy Loss 
```
for dset in 100k lt2 lt4 lt6 lt8 longtail
do 
  python3 main_sl.py -a resnet50 --cos --dist --wd 0.0005 \
  --trainset data/imagenet_${dset}_train.txt --valset data/imagenet_longtail_val.txt\
  --name ${dset}_sl_resnet50

  python3 main_sl.py -a resnet50 --cos --dist --epochs 10 \
  --pretrained ${dset}_sl_resnet50_coslr.pth.tar
  --name ${dset}2full_sl_resnet50_coslr 
done 
```

### K-positive Contrastive Learning 


#### Long-tailed Recognition 

- Representation learning 
```
python3 main_kcl.py --dist --epochs 200 --slaug --cos --wd 0.0005 --posk 6\
  --trainset data/imagenet_longtail_train.txt --valset data/imagenet_longtail_val.txt\
  --name lt_kcl6_e200
```
- Classifier learning 
```
python3 main_lincls.py -a resnet50 --dist --pretrained lt_kcl6_e200-ckpt.pth.tar\
  --trainset data/imagenet_longtail_train.txt --valset data/imagenet_longtail_val.txt\
  --name lt2lt_kcl6_e200
```

#### Out-of-Distribution Generalization 

- MoCo
```
python3 main_moco.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --clsset data/imagenet_clsset1.txt \
  path/to/imagenet

python3 main_lincls.py -a resnet50 --dist --pretrained checkpoint_0199.pth.tar \
  --clsset data/imagenet_clsset1.txt 

python3 main_lincls.py -a resnet50 --dist --pretrained checkpoint_0199.pth.tar \
  --clsset data/imagenet_clsset1_comp.txt 
```

- Supervised Learning 
```
python3 main_sl.py -a resnet50 --cos --dist --wd 0.0001 --epochs 90\
  --clsset data/imagenet_clsset1.txt --name cls1_sl

python3 main_sl.py -a resnet50 --cos --dist --wd 0.0001 --epochs 10\
  --clsset data/imagenet_clsset1_comp.txt --name cls1_sl_comp \
  --pretrained cls1_sl.pth.tar
```

- KCL 
```
python3 main_kcl.py --dist --epochs 200 --slaug --cos --wd 0.0001 --posk 6\
  --name cls1rep_kcl6_e200 --clsset data/imagenet_clsset1.txt

python3 main_lincls.py -a resnet50 --dist --pretrained cls1rep_kcl6_e200-ckpt.pth.tar\
  --name cls1_kcl6_e200 --clsset data/imagenet_clsset1.txt

python3 main_lincls.py -a resnet50 --dist --pretrained cls1rep_kcl6_e200-ckpt.pth.tar\
  --name cls1comp_kcl6_e200 --clsset data/imagenet_clsset1_comp.txt
```

### Unsupervised Training

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

To do unsupervised pre-training of a ResNet-50 model on ImageNet in an 8-gpu machine, run:
```
python main_moco.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```
This script uses all the default hyper-parameters as described in the MoCo v1 paper. To run MoCo v2, set `--mlp --moco-t 0.2 --aug-plus --cos`.

***Note***: for 4-gpu training, we recommend following the [linear lr scaling recipe](https://arxiv.org/abs/1706.02677): `--lr 0.015 --batch-size 128` with 4 gpus. We got similar results using this setting.


### Linear Classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 8-gpu machine, run:
```
python main_lincls.py \
  -a resnet50 \
  --lr 30.0 \
  --batch-size 256 \
  --pretrained [your checkpoint path]/checkpoint_0199.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```

Linear classification results on ImageNet using this repo with 8 NVIDIA V100 GPUs :
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">pre-train<br/>epochs</th>
<th valign="bottom">pre-train<br/>time</th>
<th valign="bottom">MoCo v1<br/>top-1 acc.</th>
<th valign="bottom">MoCo v2<br/>top-1 acc.</th>
<!-- TABLE BODY -->
<tr><td align="left">ResNet-50</td>
<td align="center">200</td>
<td align="center">53 hours</td>
<td align="center">60.8&plusmn;0.2</td>
<td align="center">67.5&plusmn;0.1</td>
</tr>
</tbody></table>

Here we run 5 trials (of pre-training and linear classification) and report mean&plusmn;std: the 5 results of MoCo v1 are {60.6, 60.6, 60.7, 60.9, 61.1}, and of MoCo v2 are {67.7, 67.6, 67.4, 67.6, 67.3}.


### Models

Our pre-trained ResNet-50 models can be downloaded as following:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">epochs</th>
<th valign="bottom">mlp</th>
<th valign="bottom">aug+</th>
<th valign="bottom">cos</th>
<th valign="bottom">top-1 acc.</th>
<th valign="bottom">model</th>
<th valign="bottom">md5</th>
<!-- TABLE BODY -->
<tr><td align="left"><a href="https://arxiv.org/abs/1911.05722">MoCo v1</a></td>
<td align="center">200</td>
<td align="center"></td>
<td align="center"></td>
<td align="center"></td>
<td align="center">60.6</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v1_200ep/moco_v1_200ep_pretrain.pth.tar">download</a></td>
<td align="center"><tt>b251726a</tt></td>
</tr>
<tr><td align="left"><a href="https://arxiv.org/abs/2003.04297">MoCo v2</a></td>
<td align="center">200</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center">67.7</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar">download</a></td>
<td align="center"><tt>59fd9945</tt></td>
</tr>
<tr><td align="left"><a href="https://arxiv.org/abs/2003.04297">MoCo v2</a></td>
<td align="center">800</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center">71.1</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar">download</a></td>
<td align="center"><tt>a04e12f8</tt></td>
</tr>
</tbody></table>


### Transferring to Object Detection

See [./detection](detection).


### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

### See Also
* [moco.tensorflow](https://github.com/ppwwyyxx/moco.tensorflow): A TensorFlow re-implementation.

```
pip3 install byted-dataloader==0.2.5 -i "https://bytedpypi.byted.org/simple"
```

Clear GPU
```
fuser -v /dev/nvidia*
```

### Supervised Learning
- Representation Learning for longtail

```
python3 main_sl.py -a resnet50 --dist-url 'tcp://127.0.0.1:10001' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --trainset data/imagenet_longtail_train.txt --valset data/imagenet_longtail_val.txt --cos --wd 0.0005 hdfs://haruna/home/byte_arnold_lq/user/yibairen.byron/imagenet

python3 main_sl.py -a resnet50 --trainset data/imagenet_longtail_train.txt --valset data/imagenet_longtail_val.txt --cos --wd 0.0005 --dist
```

```
python3 main_sl.py -a resnet50 --dist-url 'tcp://127.0.0.1:10001' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --trainset data/imagenet_longtail_train.txt --valset data/imagenet_longtail_val.txt --cos --pretrained ../logdir/resenet50_feat/model_best.pth.tar  --epochs 10 --wd 0.0005 --balanced --bal_repeat 4

python3 main_sl.py -a resnet50 --trainset data/imagenet_longtail_train.txt --valset data/imagenet_longtail_val.txt --cos --wd 0.0005 --epochs 10 --balanced --dist --pretrained  ../logdir/resenet50_feat/model_best.pth.tar
```