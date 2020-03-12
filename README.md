# D-DARTS: Dynamic Differentiable Architecture Search
This code is based on the official implementation of [DARTS](https://github.com/quark0/darts).


## Architecture Search
**CIFAR-10**

To carry out architecture search on CIFAR-10, run
```
python train_search_DAN.py --unrolled   # D-DARTS(2nd order)
python train_search_noise.py --unrolled # noise(2nd order)
```

**CIFAR-100**

To carry out architecture search on CIFAR-100, run
```
python train_search_DAN.py --unrolled --cifar100    # D-DARTS(2nd order)
python train_search_noise.py --unrolled --cifar100  # noise(2nd order)
```


## Architecture Train
**CIFAR-10**

To carry out architecture train on CIFAR-10, run
```
python train.py --auxiliary --cutout
```
**CIFAR-100**

To carry out architecture train on CIFAR-100, run
```
python train.py --auxiliary --cutout --cifar100
```
**ImageNet**

We used two Nvidia 2080ti(11G memory) GPUs for multi-gpu training. 
You need to manually download ImageNet on "imagenet" directory (follow the instruction [here](https://github.com/pytorch/examples/tree/master/imagenet)).
 To carry out architecture train on ImageNet, run
```
python train_imagenet.py --auxiliary --multiprocessing-distributed  
```
