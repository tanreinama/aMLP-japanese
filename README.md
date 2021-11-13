# aMLP-japanese

Japanese aMLP Pretrained Model

aMLPとは、[Liu, Daiらが提案](https://arxiv.org/abs/2105.08050)する、Transformerモデルです。

ざっくりというと、BERTの代わりに使えて、より性能の良いモデルです。

詳しい解説は、[こちら](https://ai-scholar.tech/articles/transformer/mlp_transformer)の記事などを参考にしてください。



このプロジェクトは、[スポンサーを募集しています](https://github.com/tanreinama/gpt2-japanese/blob/master/report/sponsor.md)。



# aMLP 日本語モデル

***<font color='red'>New</font>***

- 2021/11/13 - 事前学習済みbaseモデルおよびSQuADモデルを公開しました




## aMLP (Pay Attention to MLPs) とは

gMLPは、Liu, Daiらが、論文「[Pay Attention to MLPs](https://arxiv.org/abs/2105.08050)」で提案した、Self-Attention機構を排除したTransformerモデルです

BERTのTransformerモデルよりも、1層あたりのパラメーター数が少なく、その代わりに多数の層を重ねることで、同じパラメーター数あたりの性能で見て、BERTを超える性能を発揮します

ざっくりと「BERTと同じように使えてBERTより性能の良いモデル」と捉えて良いでしょう

aMLPは、gMLPにさらにSoft-Attention機構を追加することで、SQuAD等質疑応答タスクにおいてもBERTを超える性能を発揮すると報告されているモデルです

aMLP-japaneseとは、Tensorflow2で実装したaMLPモデルに、40GB超の日本語コーパスを事前学習させた、学習済みモデルです

日本語のエンコードには[Japanese-BPEEncoder_V2](https://github.com/tanreinama/Japanese-BPEEncoder_V2)を使用し、トークン数は24Kです



## TODO

✓baseモデルの公開（2021/11/13）<br>✓SQuADモデルの公開（2021/11/13）



## 公開モデル

- 事前学習モデル

| モデル名     | ダウンロードURL                                              | パラメーター数 | 学習データサイズ |
| ------------ | ------------------------------------------------------------ | -------------- | ---------------- |
| aMLP-base-ja | [https://nama.ne.jp/models/aMLP-base-ja.tar.bz2](https://nama.ne.jp/models/aMLP-base-ja.tar.bz2) （[予備URL](https://s3.ap-northeast-1.amazonaws.com/ailab.nama.ne.jp/models/aMLP-base-ja.tar.bz2)） | 67,923,648     | 40GB～           |

- 質疑応答モデル

| モデル名           | ダウンロードURL                                              | パラメーター数 | 学習データサイズ |
| ------------------ | ------------------------------------------------------------ | -------------- | ---------------- |
| aMLP-SQuAD-base-ja | [https://nama.ne.jp/models/aMLP-SQuAD-base-ja.bz2](https://nama.ne.jp/models/aMLP-SQuAD-base-ja.tar.bz2)（[予備URL](https://s3.ap-northeast-1.amazonaws.com/ailab.nama.ne.jp/models/aMLP-SQuAD-base-ja.tar.bz2)） | 67,924,674     | 200K文章         |



# 質疑応答モデル



## 使い方



GitHubからコードをクローンします

```sh
$ git https://github.com/tanreinama/aMLP-japanese
$ cd aMLP-japanese
```

学習済みモデルファイルをダウンロードして展開します

```sh
$ wget https://www.nama.ne.jp/models/aMLP-SQuAD-base-ja.tar.bz2
$ tar xvfj aMLP-SQuAD-base-ja.tar.bz2
```

以下のように「run-squad.py」を実行します

学習済みモデルを「--restore_from」に、SQuAD形式のJSONファイルを「--pred_dataset」で指定すると、質問文に対する回答が表示されます

全ての解答の候補は、「squad-predicted.json」という名前で保存されます

```sh
$ python run-squad.py --restore_from aMLP-SQuAD-base-ja --pred_dataset squad-testdata.json
Question        Answer
ロッキード・マーティン社とボーイング社が共同開発したステルス戦闘機は？  F-22戦闘機
F-22戦闘機の愛称は？    猛禽類の意味のラプター
F-22戦闘機一機あたりの価格は？  1億5千万ドル
F-22戦闘機の航続距離は？        3200km
F-22戦闘機の巡航速度は？        マッハ1.82
F-22の生産数が削減された理由は？        調達コスト
```

SQuAD型の質疑応答モデルなので、JSONファイルにコンテキストが含まれている必要があります



## ファインチューニング

ファインチューニング用の質疑応答データセットを用意して、SQuAD形式のJSONファイルで保存しておきます

そして、以下のように「run-squad.py」を実行します

学習済みモデルを「--restore_from」に、SQuAD形式のJSONファイルを「--dataset」で指定します

評価用のデータセットがあるときは、「--val_dataset」で指定すると、学習の途中で評価スコアが表示されます

```sh
$ python run-squad.py --restore_from aMLP-SQuAD-base-ja --dataset squad-testdata.json
```

一から学習させる場合は、事前学習済みモデルを「--base_model」に指定します

学習済みモデルは、「checkpoint」以下の、「--run_name」で指定したディレクトリ内に保存されます

なお、公開モデルの学習に使用した質疑応答データセットについては、著作権の関係から公開出来ません



# クラス分類モデル



## 準備



GitHubからコードをクローンします

```sh
$ git https://github.com/tanreinama/aMLP-japanese
$ cd aMLP-japanese
```

事前学習済みモデルファイルをダウンロードして展開します

```sh
$ wget https://www.nama.ne.jp/models/aMLP-base-ja.tar.bz2
$ tar xvfj aMLP-base-ja.tar.bz2
```



## 学習



クラス分類タスクでは、

```
dir/<classA>/textA.txt
dir/<classA>/textB.txt
dir/<classB>/textC.txt
・・・
```

のように、「クラス名/ファイル」という形でテキストファイルが保存されている前提で、テキストファイルをクラス毎に分類するモデルを学習します

ここでは、[livedoor ニュースコーパス](http://www.rondhuit.com/download.html#ldcc)を使用する例をサンプルとして提示します

まず、コーパスをダウンロードして展開すると、「text」以下に記事の入っているディレクトリが作成されます

```sh
$ wget https://www.rondhuit.com/download/ldcc-20140209.tar.gz
$ tar xvfz ldcc-20140209.tar.gz
$ ls text/
```

学習には、以下のように「run-classifier.py」を実行します

事前学習済みモデルを「--base_model」に、データセットのディレクトリを「--dataset」で指定します

評価用のデータセットがあるときは、「--val_dataset」で指定すると、学習の途中で評価スコアが表示されます

```sh
$ python run-classifier.py --dataset text --model aMLP-base-ja --clean_text
```

以下のようにサブディレクトリ名とクラスIDとの対応が表示された後、学習が進みます

```sh
Loading dataset...
livedoor-homme mapped index#0
kaden-channel mapped index#1
movie-enter mapped index#2
it-life-hack mapped index#3
topic-news mapped index#4
sports-watch mapped index#5
dokujo-tsushin mapped index#6
peachy mapped index#7
smax mapped index#8
```

学習済みモデルは、「checkpoint」以下の、「--run_name」で指定したディレクトリ内に保存されます



## 推論

推論には、以下のように「run-classifier.py」を実行します

学習済みモデルを「--restore_from」に、データセットのディレクトリを「--pred_dataset」、出力ファイルを「--output」で指定します

```sh
$ python run-classifier.py --pred_dataset text --output classifier-pred.csv --restore_from checkpoint/aMLP-classifier-ja/checkpoint-XXXX
$ head -n5 classifier-pred.csv
filename,pred,true
text/livedoor-homme/livedoor-homme-4956491.txt,livedoor-homme,livedoor-homme
text/livedoor-homme/livedoor-homme-5492081.txt,livedoor-homme,livedoor-homme
text/livedoor-homme/livedoor-homme-5818455.txt,livedoor-homme,livedoor-homme
text/livedoor-homme/livedoor-homme-6052744.txt,livedoor-homme,livedoor-homme
```

実行結果はCSVファイルで保存されます



# テキストの穴埋め



Masked Language Modelとして実行します。aMLPのモデルは入力されたテキスト内の「[MASK]」部分を予測します

「run-mlm.py」で、直接穴埋め問題を解かせることが出来ます

「[MASK]」一つでエンコード後のBPE一つなので、「[MASK]」が日本語1文字から3文字になります

```sh
$ python run-mlm.py --context "俺の名前は坂本[MASK]。何処にでもいるサラリー[MASK]だ。" --model aMLP-base-ja
俺の名前は坂本だ。何処にでもいるサラリーマンだ。
```



# 文章のベクトル化



[CLS]トークンに対応するベクトル表現を得ます。「--output」を指定するとファイルにカンマ区切りのテキストでファイルに保存します

```sh
$ python run-vectrize.py --context "こんにちは、世界。" --model aMLP-base-ja
[1.777146577835083, 0.5332596898078918, 0.07858406007289886, 0.5532811880111694, 0.8075544238090515, 1.3260560035705566, 0.6111544370651245, 2.338435173034668, 1.0313552618026733, ・・・
```



# REFERENCE

[Hanxiao Liu, Zihang Dai, David R. So, Quoc V. Le "Pay Attention to MLPs"  arXiv:2105.08050, 17 May 2021](https://arxiv.org/abs/2105.08050)