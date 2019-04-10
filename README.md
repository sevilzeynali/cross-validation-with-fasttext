# Cross validation with fastText
This is a program in python that uses Fasttext to do sentiment analysis for an english dataset. This program dose the cross validation to train and test fastText classifier with a simple of datasets.
We split our dataset into 10 files. we make 10 files to train and 10 to test. the texts which are in train data are not in test data. We use 10% of all dataset to test and the rest for training.
We save 10 test files in test folder, and 10 train files into train folder.
We make one model for each train file with fastText and we save the models in model folder. We test each model. We save the predictions in a preds folder. We do an average of all recals and precisions for each pair of test and train data. We display these numbers.
This program calculates the confusion matrix for your test dataset. To do this, you should change the elements of the list labels=['POS','NEG'] into your labels.
It calculates also the duration of creation of your model. The precision and recall averages for test dataset in example is 73%.
## Train and test datasets to use
To test this classifier we put to your disposal train and test datasets.
These files are some simples of files available [here](http://www.cs.cornell.edu/people/pabo/movie-review-data/). We do not any preprocessing of train or test dataset.
Our whole dataset sould be a file like this. 
```
--label--POS , These shoes are realy comfortabe. I love them.
--label__NEG , This perfum smells very bad like fake smelling flower scent. I don't like it at all.
```

## Installing and requirements
You need Python >=3.3 
You need NumPy (>= 1.8.2)
You need SciPy (>= 0.13.3)

You need to install sklearn library
```
 pip install -U scikit-learn
```
You must install [pyfasttext](https://pypi.org/project/pyfasttext/#installation).
```
pip install cython
pip install pyfasttext
```
## How to use

Usage : cross_val_with_fasttext.py
