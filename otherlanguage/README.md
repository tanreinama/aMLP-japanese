# Training with alphabet-based language

## How to run pretraining

Make working dir.

```sh
$ mkdir Lang
$ cp wiki_copy.py Lang/
```

### Getting Corpus

In example, use Hindi Language.

get C4 dataset.

```sh
$ mkdir C4; cd C4
$ GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4
$ git lfs pull --include "multilingual/c4-hi*.json.gz"
$ mv multilingual/c4-hi* ../Lang/
```

get C100 dataset

```sh
$ http://data.statmt.org/cc-100/hi.txt.xz
$ unxz hi.txt.xz; mv hi.txt Lang/
```

get wikipedia data

```sh
$ wget https://dumps.wikimedia.org/hiwiki/latest/hiwiki-latest-pages-articles.xml.bz2
$ mv hiwiki-latest-pages-articles.xml.bz2 Lang/
```

### Extract data

copy wikipedia file

```sh
$ pip install wikiextractor
$ cd Lang/
$ python -m wikiextractor.WikiExtractor hiwiki-latest-pages-articles.xml.bz2
$ python wiki_copy.py --src_dir text --dst_dir content
$ cd ..
$ python extract.py --src_dir Lang
```

make vocabulary

```sh
$ python bpe.py --src_dir Lang --language hi
$ cp Lang/vocabulary.txt hi-swe24k.txt
```

encode

```sh
$ python encoder.py --src_dir Lang/content --dst_file swe24k_encoded_hi --language hi --vocabulary hi-swe24k.txt
$ mv swe24k_encoded_hi.npz hi-swe24k.txt ../
$ cd ..
```

### Run training

```sh
$ python run-pretraining.py --dataset swe24k_encoded_hi.npz --vocabulary hi-swe24k.txt --mlm_whole_word --run_name aMLP-base-hi
```
