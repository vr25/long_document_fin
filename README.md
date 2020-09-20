Long Document Embeddings for Financial Text Regression
===============

Code and data for "Long Document Embeddings for Financial Text Regression"

## Prerequisites
This code is written in python. To use it you will need:
- Python 3.6
- [PyTorch [PyTorch] = 1.2.0](https://pytorch.org/)
- A recent version of [scikit-learn](https://scikit-learn.org/)
- A recent version of [Numpy](http://www.numpy.org)
- A recent version of [NLTK](http://www.nltk.org)
- [Tensorflow = 1.15.2](https://www.tensorflow.org)

## Getting started

The data file ```new_all_2436_mda_roa.csv``` containing all sec7/7A alongwith their corresponding ROA values and the tokenized text file ```data_2436.csv``` can be found at [Zenodo](https://zenodo.org/record/4029317#.X1-2iHXNY5k)


**To run docBERT for single task regression:**
Please note that original docBERT can be found at: 
```$ git clone https://github.com/castorini/hedwig.git``` but for this work, use modified docBERT for regression provided in this repo under directory ```hedwig```.
1. Download the dataset at: ```$ git clone https://git.uwaterloo.ca/jimmylin/hedwig-data.git```
2. Organize directory structure as:
```bash
.
+-- hedwig
|   +-- hedwig-data
```
3. Run python ```create_docbert_reg_data.py``` to create ```train.tsv```, ```dev.tsv``` and ```test.tsv``` data files from ```new_all_2436_mda_roa.csv``` and replace the files in ```/hedwig/hedwig-data/datasets/IMDB/``` with the newly generated data files.
4. Run ```python -m models.bert --dataset IMDB --model bert-base-uncased --max-seq-length 512 --batch-size 16 --lr 1e-3 --epochs 5``` for docBERT.
