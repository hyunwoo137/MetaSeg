# MetaSeg: MetaFormer-based Global Contexts-aware Network for Efficient Semantic Segmentation (WACV 2024)

### 📝[Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Kang_MetaSeg_MetaFormer-Based_Global_Contexts-Aware_Network_for_Efficient_Semantic_Segmentation_WACV_2024_paper.pdf)

> #### Beoungwoo Kang\*, Seunghun Moon\*, Yubin Cho\*, Hyunwoo Yu\*, Suk-ju Kang<sup>&dagger;</sup>
> \* Equal contribution, <sup>&dagger;</sup>Correspondence

> Sogang University

This repository contains the official Pytorch implementation of training & evaluation code for MetaSeg.

![metaseg](https://github.com/user-attachments/assets/545c2aa4-82fe-48de-951a-3e8b091a6225)

## Installation
For install and data preparation, please refer to the guidelines in [MMSegmentation v0.24.1](https://github.com/open-mmlab/mmsegmentation/blob/v0.24.1/docs/en/get_started.md#installation).

```
pip install timm
cd MetaSeg
python setup.py develop
```

## Training
Download backbone [MSCAN-T & MSCAN-B](https://cloud.tsinghua.edu.cn/d/c15b25a6745946618462/) pretrained weights.

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
<section class="section" id="BibTeX">
    <div class="container is-max-desktop content">
      <h2 class="title">Citation</h2>
      <pre><code>@inproceedings{kang2024metaseg,
  title={MetaSeg: MetaFormer-based Global Contexts-aware Network for Efficient Semantic Segmentation},
  author={Kang, Beoungwoo and Moon, Seunghun and Cho, Yubin and Yu, Hyunwoo and Kang, Suk-Ju},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={434--443},
  year={2024}
}</code></pre>
    </div>
</section>
