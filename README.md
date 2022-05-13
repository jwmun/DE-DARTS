# DE-DARTS: Dynamic Differentiable Architecture Search
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

## Citing
```
@article{mun2022darts,
  title={DE-DARTS: Neural architecture search with dynamic exploration},
  author={Mun, Jiwoo and Ha, Seokhyeon and Lee, Jungwoo},
  journal={ICT Express},
  year={2022},
  publisher={Elsevier}
}
```
## Acknowledgments

This work is in part supported by National Research Foundation of Korea (NRF, 2021R1A4A1030898(3)), Institute of Information & communications Technology Planning & Evaluation (IITP, 2021-0-00106(4)) grant funded by the Ministry of Science and ICT (MSIT), Bio-Mimetic Robot Research Center Funded by Defense Acquisition Program Administration, Agency for Defense Development (UD190018ID, 3), INMAC, and BK21-plus .

## License

MIT License