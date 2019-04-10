#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
start_time_cross_validation = time.time()
import random
import numpy as np
import glob
import os
from os.path import basename
import re
import sklearn

from pyfasttext import FastText
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#Path to text file and their labels in form : "__label__POS , text here "
LABELS_AND_TEXTS_PATH='dataset'

#path to save the models
os.mkdir("models")
MODEL_PATH='models/model'

#path to the labels estimated by each model
os.mkdir("preds")
PRED_LABELS_PATH='preds/pred'

#path to train folder
os.mkdir("train")
TRAIN_FOLDER='train/'

#path to test folder
os.mkdir("test")
TEST_FOLDER='test/'

#shuffling the data set
with open(LABELS_AND_TEXTS_PATH) as f:
    data = f.read().splitlines() 
    random.shuffle(data,random.random)

#dividing data set into train and test, we make 10 train files and 10 test files 
for k in range(0, 10):
  test_data = data[int(float(k)/10*len(data)):int(float(k+1)/10*len(data))] #Divides 1/10th of the data as test data
  train_data = [x for x in data if x not in test_data] #Divides 9/10th of the data as training data
  
  #save test files 
  with open(TEST_FOLDER+str(k),"w") as test_file:
    for test in test_data:
      test_file.write(test+"\n")
  
  #save train files
  with open(TRAIN_FOLDER+str(k),"w") as train_file:
    for train in train_data:
      train_file.write(train+"\n")

for train in glob.glob(TRAIN_FOLDER+"*"):
  print("Processing of "+basename(train))  
  print("Processing of the model")
  
  #creating model with fastText
  model=FastText()
  classifier = model.supervised(input=train,output=MODEL_PATH+basename(train),lr=0.02,epoch=50)
  print("Testing the model")
  
  #testing the model
  print(TEST_FOLDER+basename(train))
  result=model.predict_file(TEST_FOLDER+basename(train), k=1)
  
  #opening test file to do confusion matrix
  with open(TEST_FOLDER+basename(train)) as f:
    test_dataset=f.read().splitlines()

    pred_labels_list=[]# list to save predicted labels with each model 
    test_labels=[]#list to save the real labels
    test_labels_normalized=[]#list to save real labels in forme  label1\nlabel2\nlabel3

    #opening test files to save labels 
    pred_labels=[] #list of list of labels estimated by the model
    with open(TEST_FOLDER+basename(train)) as f:
      for row in f:
        test_labels.append(row.split()[0])  
    #doing a list with real labels for the confusion matrix
    for labels in test_labels:
        labels=re.sub('__label__','',labels)
        test_labels_normalized.append(labels)
    #estimating labels 
    pred_labels=model.predict_file((TEST_FOLDER+basename(train)))
    #writing estimated labels in a file
    write_labels=open(PRED_LABELS_PATH+basename(train),'w')
    for element in pred_labels:
      for label in element:
        write_labels.write(label+"\n")
        pred_labels_list.append(label)
    print("I do the confusion matrix")
    print(confusion_matrix(test_labels_normalized, pred_labels_list,labels=['NEG','POS']))
    #précision,rappel et f-score et support pour chaque classe
    av2=sklearn.metrics.classification_report(test_labels_normalized, pred_labels_list,target_names=['NEG','POS'])
    print(av2)
    

      
