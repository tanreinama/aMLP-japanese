# aMLP-japanese-PyTorch

Japanese aMLP Pretrained Model for PyTorch Japanese / [English](README-en.md)

aMLPとは、[Liu, Daiらが提案](https://arxiv.org/abs/2105.08050)する、Transformerモデルです。

ざっくりというと、BERTの代わりに使えて、より性能の良いモデルです。

詳しい解説は、[こちら](https://ai-scholar.tech/articles/transformer/mlp_transformer)の記事などを参考にしてください。


# PyTorch版



PyTorch版のモデルは、TensorFlow版のモデルを変換して作成します。

現在、学習済みモデルの変換プログラムと、SQuADタスクの実行コードがあります。

モデルの変換は「tfmodel2torch.py」で行います。

```sh
$ cd pytorch
$ python tfmodel2torch.py --tf_model_dir ../aMLP-SQuAD-base-ja --output aMLP-SQuAD-base-ja.pt
has_voc: True is_classifier: False is_squad: True
$ ls aMLP-SQuAD-base-ja.*
aMLP-SQuAD-base-ja.json  aMLP-SQuAD-base-ja.pt  aMLP-SQuAD-base-ja.txt
```

SQuADタスクの実行は、「test-squad.py」で行います。

全ての解答の候補は、「squad-predicted.json」という名前で保存されます

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
