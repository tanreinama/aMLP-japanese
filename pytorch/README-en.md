# aMLP-japanese-PyTorch

Japanese aMLP Pretrained Model for PyTorch [Japanese](README.md) / English

aMLP is a Transformer model proposed by [Liu, Dai et al.](Https://arxiv.org/abs/2105.08050).

Roughly speaking, it is a model with better performance that can be used instead of BERT.


# PyTorch Version



The PyTorch version model is created by converting the TensorFlow version model.

Currently, I have a transform program for the trained model and the execution code for the SQuAD task.

Model conversion program is "tfmodel2torch.py".

```sh
$ cd pytorch
$ python tfmodel2torch.py --tf_model_dir ../aMLP-SQuAD-base-ja --output aMLP-SQuAD-base-ja.pt
has_voc: True is_classifier: False is_squad: True
$ ls aMLP-SQuAD-base-ja.*
aMLP-SQuAD-base-ja.json  aMLP-SQuAD-base-ja.pt  aMLP-SQuAD-base-ja.txt
```

The SQuAD task is executed in "test-squad.py".

All answer suggestions are saved as "squad-predicted.json"

```sh
$ python test-squad.py --pred_dataset ../squad-testdata.json --model aMLP-SQuAD-base-ja.pt
Question        Answer
ロッキード・マーティン社とボーイング社が共同開発したステルス戦闘機は？  F-22戦闘機
F-22戦闘機の愛称は？    猛禽類の意味のラプター
F-22戦闘機一機あたりの価格は？  1億5千万ドル
F-22戦闘機の航続距離は？        3200km
F-22戦闘機の巡航速度は？        マッハ1.82
F-22の生産数が削減された理由は？        調達コスト
```



# REFERENCE

[Hanxiao Liu, Zihang Dai, David R. So, Quoc V. Le "Pay Attention to MLPs"  arXiv:2105.08050, 17 May 2021](https://arxiv.org/abs/2105.08050)
