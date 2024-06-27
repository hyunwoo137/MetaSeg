# MetaSeg: MetaFormer-based Global Contexts-aware Network for Efficient Semantic Segmentation

The code is based on [MMSegmentaion v0.24.1](https://github.com/open-mmlab/mmsegmentation/tree/v0.24.1).

## Installation
For install and data preparation, please refer to the guidelines in [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/blob/v0.24.1/docs/en/get_started.md#installation).

```
pip install timm
cd MetaSeg
python setup.py develop
```

## Training
Download backbone [ MSCAN-T & MSCAN-B ] pretrained weights.

Put them in a folder ```pretrain/```.

Example - Train ```MetaSeg-T``` on ```ADE20K```:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh local_configs/metaseg/tiny/metaseg.tiny.512x512.ade.160k.py <GPU_NUM>
```

## Evaluation
Download trained weights.

Example - Evaluate ```MetaSeg-T``` on ```ADE20K```:

```
# Single-gpu testing
CUDA_VISIBLE_DEVICES=0 python tools/test.py local_configs/metaseg/tiny/metaseg.tiny.512x512.ade.160k.py /path/to/checkpoint_file

# Multi-gpu testing
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_test.sh local_configs/metaseg/tiny/metaseg.tiny.512x512.ade.160k.py /path/to/checkpoint_file <GPU_NUM>
```
