# [ICMR21] Few-Shot Action Localization without Knowing Boundaries

![arch](asset/overview.pdf)

### Link: 

[[Arxiv]](https://arxiv.org/abs/2106.04150)
[[Project]](https://github.com/June01/WFSAL-icmr21)
[[Presentation]]()

If you find this helps your research, please cite:

```
@article{Xie2021FewShotAL,
  title={Few-Shot Action Localization without Knowing Boundaries},
  author={Tingting Xie and Christos Tzelepis and Fan Fu and Ioannis Patras},
  journal={Proceedings of the 2021 International Conference on Multimedia Retrieval},
  year={2021}
}
```

---
## Contents

* [Install](#install)
  
* [Download](#download)

* [Training](#training)

* [Use reference models for evaluation](#use-reference-models-for-evaluation)

* [Without Learning](#no-learning)

* [Related project](#related-project)

* [Contact](#contact)

--- 

### Install

```
git clone https://github.com/June01/WFSAL-icmr21
cd WFSAL-icmr21
pip install -r requirements.txt
```

### Download 

Please first create a data dir and then put all the features and annotations under it.
```
mkdir data
cd data
```
The feature and annotations used in this paper are originated from [wtalc](https://github.com/sujoyp/wtalc-pytorch). 
The features for Thumos14 and ActivityNet1.2 dataset can be downloaded [here](https://emailucr-my.sharepoint.com/:f:/g/personal/sujoy_paul_email_ucr_edu/Es1zbHQY4PxKhUkdgvWHtU0BK-_yugaSjXK84kWsB0XD0w?e=I836Fl), while annotations can be found in the original repo.

### Training (5-way 1-shot)

For thumos 14,

```
python main.py --split='cvpr18' --encoder --num_in=4 --tsm='ip' --sample_num_per_class=1 --batch_num_per_class=5 
```

For ActivityNet1.2:
```
python main.py --split='cvpr18' --dataset='ActivityNet1.2' --num_in=4 --encoder --tsm='ip' --sample_num_per_class=1
```

For evaluation,
```
python main.py --split='cvpr18' --dataset=dataset --num_in=4 --encoder --tsm=ip --sample_num_per_class=1 --mode=testing --load=/path/to/model
```
Note, we report the median of 10 repetitions.

### Without Learning

```
python eval_non_learning.py
```

### Related project

- [wtalc-pytorch](https://github.com/sujoyp/wtalc-pytorch#readme)

- [3c-net](https://github.com/naraysa/3c-net)

- [DGAM-Weakly-Supervised-Action-Localization](https://github.com/bfshi/DGAM-Weakly-Supervised-Action-Localization)

### Contact

For any question, please file an issue or contact

```
t.xie@qmul.ac.uk
```

