# NQE

Official resources of **"NQE: N-ary Query Embedding for Complex Query Answering over Hyper-relational Knowledge Graphs"**. Haoran Luo, Haihong E, Yuhao Yang, Gengxian Zhou, Yikai Guo, Tianyu Yao, Zichen Tang, Xueyuan Lin, Kaiyang Wan. **AAAI 2023** \[[paper](https://doi.org/10.48550/arXiv.2211.13469)\]. 

**Please also star the main code contributor [Yuhao Yang](https://github.com/TimeFighter818/NQE_Nary_Query_Embedding) from BUAA.**

## Overview
An example of n-ary FOL query:
![](./figs/F1.drawio.png)

16 kinds of n-ary FOL queries in WD50K-NFOL:
![](./figs/F4.drawio.png)

## Setup

### Default implementation environment

* *Linux(SSH) + Python3.7.13 + Pytorch1.8.1 + Cuda10.2*

```
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

### Install Dependencies

Install dependencies:
```
pip install -r requirements.txt
```


### Configure the Dataset

We tested the effectiveness of our model on two datasets, including the WD50K-QE dataset and the WD50K-NFOL dataset.

* WD50K-QE dataset is a dataset created with the multi-hop reasoning method StarQE. It covering multi-hop reasoning with conjunction logic operation. We call it "wd50k_qe" in the code. You can download and upzip the WD50K-QE dataset file from [TeraBox](https://terabox.com/s/1jiIHls_9FSoY--ULOMpcOg) or [Baidu Netdisk](https://pan.baidu.com/s/1IIFaPfXQIKKdWbX1HX6heA?pwd=26mp).

```
unzip wd50k_qe.zip -d data/
```

* WD50K-NFOL is a hyper-relational dataset we created covering logical operations including conjunction, disjunction and negation as well as combined operations. We call it "wd50k_nfol" in the code. You can download and upzip the WD50K-NFOL dataset file from [TeraBox](https://terabox.com/s/1jiIHls_9FSoY--ULOMpcOg) or [Baidu Netdisk](https://pan.baidu.com/s/1WVkyDQbL4l73jd1IZt7q_Q?pwd=5inx).

```
unzip wd50k_nfol.zip -d data/
```

### Generate the Groundtruth ###

Then, we should generate the groundtruth of the chosen dataset for evaluation. If you don't change the dataset, please skip this step, because the zip files above have already got the groundtruth in "gt\" file by following operation.

* For WD50K-QE dataset:
```
python src/generate_groundtrurh.py --dataset wd50k_qe
```
* For WD50K-NFOL dataset:
```
python src/generate_groundtrurh.py --dataset wd50k_nfol
```

## Model Training ##

You can train query embedding model using "src/map_iter_qe.py".

* For WD50K-QE dataset:
```
python src/map_iter_qe.py --dataset wd50k_qe --epoch 300 --gpu_index 0
```

* For WD50K-NFOL dataset:
```
python src/map_iter_qe.py --dataset wd50k_nfol --epoch 30 --gpu_index 0
```

## Evaluation ##

You can only run prediction using "src/map_iter_qe.py" by with argument "do_learn" False and argument "do_predict" True.

In this case, you need to select the ckpts file you want to use and configure the "prediction_ckpt" argument as you want.

* For WD50K-QE dataset:
```
python src/map_iter_qe.py --dataset wd50k_qe --do_learn False --do_predict True --prediction_ckpt ckptswd50k_qe-train_tasks-1p-best-valid-DIM256.ckpt --prediction_tasks 1p,2p,3p,2i,3i,pi,ip --gpu_index 0
```

* For WD50K-NFOL dataset:
```
python src/map_iter_qe.py --dataset wd50k_nfol --do_learn False --do_predict True --prediction_ckpt ckptswd50k_nfol-train_tasks-1p-best-valid-DIM256.ckpt --prediction_tasks 1p,2p,3p,2i,3i,pi,ip,2u,up,2cp,3cp,2in,3in,inp,pin,pni --gpu_index 0
```

## BibTex

If you find this work is helpful for your research, please cite:

```bibtex
@article{luo2022nqe,
  title={NQE: N-ary Query Embedding for Complex Query Answering over Hyper-relational Knowledge Graphs},
  author={Luo, Haoran and E, Haihong and Yang, Yuhao and Zhou, Gengxian and Guo, Yikai and Yao, Tianyu and Tang, Zichen and Lin, Xueyuan and Wan, Kaiyang},
  journal={arXiv preprint arXiv:2211.13469},
  year={2022}
}
```

For further questions, please contact: luohaoran@bupt.edu.cn, or wechat: lhr1846205978.