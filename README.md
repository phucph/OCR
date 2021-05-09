# PixelLink

This code is based on PixlLink and PSENet, the performance is not satisfactory

## Requirements
* Python 3.6
* PyTorch v0.4.1+
* opencv-python 3.4


## Training
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ic15.py --arch vgg16 --batch_size 24
```

## Testing
```
CUDA_VISIBLE_DEVICES=0 python test_ic15.py --resume [path of model]
```

## Eval script for ICDAR 2015
```
cd eval
sh eval_ic15.sh
```


## Performance
| Dataset | Pretrained | Precision (%) | Recall (%) | F-measure (%) | FPS (1080Ti) | Input |
| - | - | - | - | - | - | - |
| ICDAR2015 | No | 81.2 | 75 | 78 | 5 | 1280*768 |

## TODO
- [ ] Find the bug of the performance issue.
- [ ] Accomplish the code with better config file and more datasets
