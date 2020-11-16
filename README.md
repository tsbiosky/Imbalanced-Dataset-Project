# Imbalanced-Dataset-Project



## Preparing
Prepare your data before training. The format of your data should follow the file in `datasets`.
You can downlaod Cityscapes dataset and run </br>  `python get_bbox_seg.py`</br>
or download processed dataset [here](https://drive.google.com/file/d/1mSYC_drMkVwapKPLJaMSJ-sloFiBTMLm/view?usp=sharing)
## Training stage
```bash
python train.py --dataroot ./dataset --name model_name --model seg2pix --which_model_netG unet_256  --lambda_A 100 --dataset_mode aligned --use_spp --no_lsgan --norm batch
```
[Pretrained checkpoints](https://drive.google.com/file/d/1Qqevam5H5sClF04nXtEJaqYRfoW71_c1/view?usp=sharing) put it under /checkpoints/model_name/
## Testing stage
```bash
python test.py --dataroot data_path --name model_name --model seg2pix --which_model_netG unet_256   --dataset_mode aligned --use_spp --norm batch
```
## Data augmentation stage
The results are under folder aug_pedestrian.
Follow the instruction in  <a href="https://github.com/open-mmlab/mmdetection">MMdetection</a> to convert Cityscapes to COCO fomat.
Then  run</br> `python data_aug.py`
![image](https://github.com/tsbiosky/Imbalanced-Dataset-Project/blob/master/seg2pix.png)
## Evaluation stage
Put the   new Cityscapes dataset under  mmdection folder .Follow the instruction in  <a href="https://github.com/open-mmlab/mmdetection">MMdetection</a>  to train a object detector and evaluate it 

## Acknowledgments
Heavily borrow the code from <a href="https://github.com/yueruchen/Pedestrian-Synthesis-GAN">PSGAN</a>

