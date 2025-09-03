# ILNet: Trajectory Prediction With Inverse Learning Attention for Enhancing Intention Capture

This is the ILNet official implementation.
This paper is accepted for publication in the IEEE Transactions on Intelligent Transportation Systems (T-ITS), 2025.

[arXiv](https://arxiv.org/abs/2507.06531) | [paper](https://ieeexplore.ieee.org/abstract/document/11130613)


## Table of Contents
+ [Setup](#setup)
+ [Training](#training)
+ [Validation](#validation)
+ [Acknowledgements](#acknowledgements)
+ [Citation](#citation)

## Setup
[Follow the HPNet settings](https://github.com/XiaolongTang23/HPNet)


## Training

```
# For Argoverse
python ILNet-Argoverse/train.py --root /path/to/Argoverse_root/ --train_batch_size 3 --val_batch_size 3 --devices 4

# For INTERACTION
python ILNet-INTERACTION/train.py --root /path/to/INTERACTION_root/ --train_batch_size 4 --val_batch_size 4 --devices 4
```

## Validation
```
# For Argoverse
python ILNet-Argoverse/val.py --root /path/to/Argoverse_root/ --val_batch_size 3 --devices 4 --ckpt_path /path/to/checkpoint.ckpt

# For INTERACTION
python ILNet-INTERACTION/val.py --root /path/to/INTERACTION_root/ --val_batch_size 4 --devices 4 --ckpt_path /path/to/checkpoint.ckpt
```


## Acknowledgements
We sincerely appreciate [Argoverse](https://github.com/argoverse/argoverse-api), [INTERACTION](https://github.com/interaction-dataset/interaction-dataset),[HPNet](https://github.com/XiaolongTang23/HPNet) for their awesome codebases.


## Citation

If ILNet has been helpful in your research, please consider citing our work:

```
@article{zeng2025ilnet,
  title={ILNet: Trajectory Prediction With Inverse Learning Attention for Enhancing Intention Capture},
  author={Zeng, Mingjin and Ouyang, Nan and Wan, Wenkang and Ao, Lei and Cai, Qing and Sheng, Kai},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2025},
  publisher={IEEE}
}
```
