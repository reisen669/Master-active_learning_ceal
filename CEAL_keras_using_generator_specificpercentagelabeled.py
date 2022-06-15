"""
source: https://github.com/dhaalves/CEAL_keras

reference:
- https://keras.io/api/applications/resnet/#resnet50-function
- https://www.codegrepper.com/code-examples/python/tensorflow+ignore+warnings
- https://superuser.com/questions/283673/how-to-get-windows-command-prompt-to-display-time
- https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/

setup:
- reset all training data to be unlabeled (no validation)
- split into train + validation datasets
- run 
 
added:
- runtime display
- loss and accuracy graphs
- classification report
- save best model
- classification report in csv

note:
- no pretrained weights
- model used affects image size
- dataset used 
- entropy calculation generates runtime warning
- each AL iteration not initial model; trained model carried forward

questions:
- validation data? train-test-val?
- write labeled and unlabeled image filepaths to textfile?
- keep loading images from file when labeled data updated?
- always move labeled data?

notessssss:
+ Microsoft Visual Studio
+ TF 
+ ori used CIFAR10 dataset, modified to work on custom dataset
+ ori loaded all data into memory, modified to load into memory in batch

"""
import argparse
import os

# ignore warnings and debug/logging 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf     # current: 2.4.0
# import logging
# logger = logging.getLogger()
# tensorflow.get_logger().setLevel("ERROR")   
#logging.setLevel("ERROR")   

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
# from keras.datasets import cifar10
from tensorflow.keras.models import load_model  # keras.models
# from keras.utils import np_utils
# from keras_contrib.applications.resnet import ResNet18
from tensorflow.keras.applications import ResNet50V2 #ResNet50
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB7, EfficientNetB6     #efficientnet
# from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import BinaryAccuracy, FalseNegatives, FalsePositives

import shutil
import random
import time
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import pandas as pd
import math
import gc
from PIL import features

FOLD_TEXTFILE = './data/extended_fold_4.txt' # extended txt filename
TESTFOLD = '_4'  #_4_extended
TESTFOLD_TEST = '_4' #_4_extended

TRAIN_DIR = f'./data/train{TESTFOLD}/pool'    #f'./data/train{TESTFOLD}/pool' #my_NSFW_dataset/train/pool
TRAIN_HC_DIR = f'./data/train{TESTFOLD}/hc'  #f'./data/train{TESTFOLD}/hc'
TRAIN_LABELED_DIR = f'./data/train{TESTFOLD}/labeled'    #f'./data/train{TESTFOLD}/labeled'
TRAIN_VAL_DIR = f'./data/train{TESTFOLD}/val' #f'./data/train{TESTFOLD}/val'
TEST_DIR = f'./data/test{TESTFOLD_TEST}'     #f'./data/test{TESTFOLD_TEST}' #my_NSFW_dataset/test   # used as validation data
        # 0              1                  2               3                   4           
CNN = ['ResNet50_v2', 'EfficientNetB7', 'EfficientNetB0', 'EfficientNetB6', 'Inception']
CNN_MODEL = CNN[0]

# set layer
LAYER = 'GlobalAveragePool' #'Flatten', 'GlobalAveragePool'

if CNN_MODEL == CNN[0]:
    from tensorflow.keras.applications.resnet_v2 import preprocess_input
elif CNN_MODEL == CNN[1] or CNN_MODEL == CNN[2] or CNN_MODEL == CNN[3]:
    from tensorflow.keras.applications.efficientnet import preprocess_input #inception

# set input size
if CNN_MODEL == CNN[1]:
    DIM = 600
if CNN_MODEL == CNN[3]:
    DIM = 528
elif CNN_MODEL == CNN[0] or CNN_MODEL == CNN[2]:
    DIM = 224


CLASSES = ['NonPorn', 'Porn']
#CLASSES = ['SFW', 'NSFW']
RESET = True
RESET_VAL_DATA = False       # val set
DISPLAY_ONE_BATCH = False
RESET_ONLY = False           # all data including test set, val set depends on the other setting


''' deprecated'''
"""
# https://stackoverflow.com/questions/49404993/how-to-use-fit-generator-with-multiple-inputs
# https://github.com/keras-team/keras/issues/9969
class MultipleInputGenerator(Sequence):
    #Wrapper of 2 ImageDataGenerator
    def __init__(self, seq1, seq2):
        self.seq1, self.seq2 = seq1, seq2
        self.toggle = True
    
    def __len__(self):
        return len(self.seq1) 

    def __getitem__(self, idx):
        x1, y1 = self.seq1[idx]
        x2, y2 = self.seq2[idx]
        return [x1, x2], [y1, y2]

def initialize_cifar_dataset():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data() # 50k, 10k

    n_classes = np.max(y_test) + 1

    # Convert class vectors to binary class matrices.
    y_train = np_utils.to_categorical(y_train, n_classes)
    y_test = np_utils.to_categorical(y_test, n_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    # # subtract mean and normalize [ResNet18]
    # mean_image = np.mean(x_train, axis=0)
    # x_train -= mean_image
    # x_test -= mean_image
    # x_train /= 128.
    # x_test /= 128.

    x_train, x_test = preprocess_input(x_train), preprocess_input(x_test) # [ResNet50, ResNet50V2]
    
    initial_train_size = int(x_train.shape[0] * args.initial_annotated_perc) # 5000
    x_pool, x_initial, y_pool, y_initial = train_test_split(x_train, y_train, test_size=initial_train_size,
                                                            random_state=1, stratify=y_train)

    return x_pool, y_pool, x_initial, y_initial, x_test, y_test, n_classes

def refresh_unlabeled_hc_generators(train_datagen, train_dir, seed):
    # update unlabeled and high confidence data
    train_pool_gen = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(DIM,DIM),  # resize
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        class_mode='categorical',   # e.g., [0. 1.]
                                                        color_mode='rgb',
                                                        seed=seed)
    train_hc_gen = train_datagen.flow_from_directory(train_dir.replace('pool','hc'),
                                                    target_size=(DIM,DIM),  # resize
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    class_mode='categorical',   # e.g., [0. 1.]
                                                    color_mode='rgb',
                                                    seed=seed)
    print('UPDATED: ', train_pool_gen.samples, train_hc_gen.samples)
    
    # lists of training (unlabeled and high confidence) image filepaths
    x_fp_pool, x_fp_hc = train_pool_gen.filenames, train_hc_gen.filenames
    _, pool_counts = np.unique(train_pool_gen.labels, return_counts=True)
    _, hc_counts = np.unique(train_hc_gen.labels, return_counts=True)
    print(f'{pool_counts} in unlabeled, {hc_counts} in high confidence')
    return train_pool_gen, train_hc_gen, x_fp_pool, x_fp_hc, pool_counts
"""

''' Initialisation '''
def show_batch(image_batch, label_batch):
        plt.figure(figsize=(10,10))
        for i, (im, label) in enumerate(zip(image_batch, label_batch)):
            print(i, label, CLASSES[np.argmax(label)])
            ax = plt.subplot(6,6,i+1)
            plt.imshow(im)
            plt.title(CLASSES[np.argmax(label,axis=0)]) #CLASS_NAMES[label_batch[n]==1][0].title()
            plt.axis('off')
        plt.close()
        plt.show()
                            
def reset_original_data(x_fp_labeled): # x_fp_hc
    # validation data
    if RESET_VAL_DATA:
        for c in CLASSES:
            for img in os.listdir(os.path.join(TRAIN_VAL_DIR, c, 'true')):
                shutil.move(os.path.join(TRAIN_VAL_DIR, c, 'true', img), 
                            os.path.join(TRAIN_DIR, c, 'true', img))
    
    # labeled data
    for fp in x_fp_labeled:
        if 'hc' in fp.split("\\"):
            continue
        shutil.move(os.path.join(TRAIN_LABELED_DIR, fp), 
                    os.path.join(TRAIN_DIR, fp))
                    
    # labeled data
    for fp in x_fp_labeled:
        if 'hc' in fp.split("\\"):
            continue
        shutil.move(os.path.join(TRAIN_LABELED_DIR, fp), 
                    os.path.join(TRAIN_DIR, fp))
    
    # high confidence data
    if os.path.exists(TRAIN_HC_DIR):
        for c in CLASSES:
            for var in os.listdir(os.path.join(TRAIN_HC_DIR, c)):
                if os.path.isdir(os.path.join(TRAIN_HC_DIR, c, var)):
                    for fp in os.listdir(os.path.join(TRAIN_HC_DIR, c, var)):
                        shutil.move(os.path.join(TRAIN_HC_DIR, c, var, fp), 
                                    os.path.join(TRAIN_DIR, c, var, fp))
                else:
                    shutil.move(os.path.join(TRAIN_HC_DIR, c, var), 
                                os.path.join(TRAIN_DIR, c, var))
    
    # remove duplicate high confidence data
    try:
        for c in CLASSES:
            shutil.rmtree(os.path.join(TRAIN_LABELED_DIR, c, 'hc'))
        shutil.rmtree(os.path.join(TRAIN_HC_DIR)) 
    except:
        print('Skipped deletion')

def extract_images_from_subfolders():
    for c in CLASSES:
        curr = os.path.join(TRAIN_DIR, c, 'true')
        for folder in os.listdir(curr):
            # extract files from folder if folder exists
            if os.path.isdir(os.path.join(curr, folder)):
                for img in os.listdir(os.path.join(curr, folder)):
                    shutil.move(os.path.join(curr, folder, img), os.path.join(curr, img))
                shutil.rmtree(os.path.join(curr, folder))
    print('Images extracted from subfolders')

def manual_split_train_val_dataset(seed, val_split):
    # return if validation data already exists
    if (len(os.listdir(os.path.join(TRAIN_VAL_DIR, CLASSES[0], 'true'))) != 0) or \
            (len(os.listdir(os.path.join(TRAIN_VAL_DIR, CLASSES[1], 'true'))) != 0):
        if (len(os.listdir(os.path.join(TRAIN_VAL_DIR, CLASSES[0], 'true'))) != 0) and \
                (len(os.listdir(os.path.join(TRAIN_VAL_DIR, CLASSES[1], 'true'))) != 0):
            print('Use existing validation data\n')
            return
        else:
            print('Error in spliting validation data!')
            exit(0)
            
    # else, split unlabeled data
    # move validation data into corresponding folds if textfiles are available (cross val)
    if any(file.endswith('.txt') for file in os.listdir(TRAIN_VAL_DIR)):
        for file in [fn for fn in os.listdir(TRAIN_VAL_DIR) if fn.endswith('.txt')]: 
            foldname = file.split('_')[1].split('.')[0]
            #print(file, foldname)
            
            # create folder if required
            curr = os.path.join(f'{TRAIN_VAL_DIR}_{foldname}') 
            if not os.path.exists(curr):
                os.makedirs(curr)    
            
            for c in CLASSES:
                if not os.path.exists(os.path.join(curr, c)):
                    os.makedirs(os.path.join(curr, c))
                
                if not os.path.exists(os.path.join(curr, c, 'true')):
                    os.makedirs(os.path.join(curr, c, 'true'))
            
            # move validation data accordingly
            with open(os.path.join(TRAIN_VAL_DIR, file), 'r') as f:
                lines = f.readlines()
                for l in lines:
                    l = l.strip('\n')
                    shutil.move(os.path.join(l), os.path.join(curr, l.split(TRAIN_DIR)[1][1:]))
        print('Validation dataset set up')
        
        full_data = []
        for c in CLASSES:
            for folder in os.listdir(os.path.join(TRAIN_DIR, c)):
                for img in os.listdir(os.path.join(TRAIN_DIR, c, folder)):
                    full_data.append(os.path.join(TRAIN_DIR, c, folder, img))
        # shuffle whole list and split into folds
        random.shuffle(full_data)
        folds = [full_data[x:x+target_num] for x in range(0, len(full_data), target_num)]
        for i, fold in enumerate(folds):
            write_val_data_to_textfile(val_list=fold, filename=os.path.join(TRAIN_VAL_DIR, f'val_{i}.txt'))
                
            # for val_data in val_list:
                # shutil.move(os.path.join(val_data), os.path.join(TRAIN_VAL_DIR, val_data.split(TRAIN_DIR)[1][1:]))
        print('Validation dataset split into folds (txt form)')
    else:
        print('No textfile for other validation fold')
        
        full_data = []
        for c in CLASSES:
            for folder in os.listdir(os.path.join(TRAIN_DIR, c)):
                for img in os.listdir(os.path.join(TRAIN_DIR, c, folder)):
                    full_data.append(os.path.join(TRAIN_DIR, c, folder, img))
        
        target_num = int(len(full_data)*val_split)
        #/*
        sampled_num = 0
        while sampled_num < target_num:
            val_list = random.sample(full_data, (target_num - sampled_num))
            #print(len(full_data), len(val_list))
            # # check if sampled data already exist in other folds
            # try:
                # for file in [fn for fn in os.listdir(TRAIN_VAL_DIR) if fn.endswith('.txt')]: 
                    # print(file)
                    # other_list = []
                    # with open(os.path.join(TRAIN_VAL_DIR, file), 'r') as f:
                        # lines = f.readlines()
                        # for l in lines:
                            # other_list.append(l.strip('\n'))
                    # for val_data
            # except:
                # print('No textfile for other validation fold')
            # #*/
            
        # shuffle whole list and split into folds
        # random.shuffle(full_data)
        # folds = [full_data[x:x+target_num] for x in range(0, len(full_data), target_num)]
        # for i, fold in enumerate(folds):
            # write_val_data_to_textfile(val_list=fold, filename=os.path.join(TRAIN_VAL_DIR, f'val_{i}.txt'))
             
            for val_data in val_list:
                try:   
                    shutil.move(os.path.join(val_data), os.path.join(TRAIN_VAL_DIR, val_data.split(TRAIN_DIR)[1][1:]))
                    sampled_num +=1
                except e as Exception:
                    print(f'{val_data} not found ({sampled_num})\n{e}')
        print('Validation dataset split into folds')
    
    exit(0)

def random_create_test_dataset(test_split=0.25):
    # return number of existing test data 
    existing_num = 0
    for c in CLASSES:
        existing_num += len(os.listdir(os.path.join(TEST_DIR, c)))
    
    # move randomly selected unlabeled data to test set
    full_data = []
    for c in CLASSES:
        for folder in os.listdir(os.path.join(TRAIN_DIR, c)):
            for img in os.listdir(os.path.join(TRAIN_DIR, c, folder)):
                full_data.append(os.path.join(TRAIN_DIR, c, folder, img))
    
    test_list = random.sample(full_data, (int((len(full_data) + existing_num) * val_split) - existing_num))
    print(len(full_data), len(test_list))
    print(f'{existing_num} test data already exists ' + \
          f'[{int(len(full_data)*val_split) - existing_num} + {existing_num} = {int(len(full_data)*val_split)}]')
    
    for test_data in test_list:
        folders = test_data.split(TRAIN_DIR)[1].split('\\')
        print(folders)
        new_fp = os.path.join(TEST_DIR, folders[0], folders[2])
        print(new_fp)
        exit(0)
        shutil.move(os.path.join(test_data), new_fp)
    print(f'Test dataset set up')
    exit(0)

def move_fold_test_data(filename):
    full_data = []
    # read textfile
    with open(os.path.join(filename), 'r') as f:
        lines = f.readlines()
        for l in lines:
            full_data.append(l.strip('\n'))
    print(len(full_data))
    
    # move unlabeled data in textfile to test set
    for x in full_data:
        subs = x.split(TRAIN_DIR)[1].split('/')
        #print(os.path.join(TEST_DIR, subs[1], subs[-1]))
        shutil.move(os.path.join(x), os.path.join(TEST_DIR, subs[1], subs[-1]))
    
    total_num = 0
    for c in CLASSES:
        total_num += len(os.listdir(os.path.join(TEST_DIR, c)))
        
    print(f'Test dataset set up [{total_num}]')
    exit(0)
     
def initialize_custom_dataset(seed, sample_size):
    train_datagen = ImageDataGenerator(horizontal_flip = True, vertical_flip = True, 
                                       width_shift_range = 0.1, height_shift_range = 0.1,
                                       channel_shift_range=0, 
                                       zoom_range = 0.2, rotation_range = 20,
                                       preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    # update labeled and unlabeled data in DirectoryIterator
    train_pool_gen, train_labeled_gen, x_fp_pool, x_fp_labeled, pool_counts = \
            refresh_unlabeled_labeled_generators(train_datagen, seed)
    val_generator = train_datagen.flow_from_directory(TRAIN_VAL_DIR,
                                                    target_size=(DIM,DIM),
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    class_mode='categorical',  
                                                    color_mode='rgb',
                                                    seed=seed)
    test_generator = test_datagen.flow_from_directory(TEST_DIR,
                                                    target_size=(DIM,DIM),
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    class_mode='categorical',  
                                                    color_mode='rgb',
                                                    seed=seed)
    print(f'{train_pool_gen.samples} unlabeled   {train_labeled_gen.samples} labeled', '\n') #train_pool_gen.class_indices
    
    # inspect a batch
    if DISPLAY_ONE_BATCH:
        image_batch, label_batch = next(train_pool_gen)  # image data as float32 data
        show_batch(image_batch, label_batch)
    
    # reset data
    if RESET or RESET_ONLY:
        reset_original_data(x_fp_labeled=x_fp_labeled) 
        if RESET_ONLY:
            if 'extended' in TESTFOLD: # prepare test set
                move_fold_test_data(filename=FOLD_TEXTFILE)
                exit(0)
        # exit(0)
        train_pool_gen, train_labeled_gen, x_fp_pool, x_fp_labeled, pool_counts = \
                refresh_unlabeled_labeled_generators(train_datagen, seed, True)
        print(f'Settings reset: {train_pool_gen.samples} unlabeled   {train_labeled_gen.samples} labeled', '\n')
        
    if RESET: 
        # extract randomly selected initial training data as labeled
        count_np, count_p = 0, 0
        ind_list = [val for val in range(len(x_fp_pool))]
        while len(ind_list) > 0 and (count_np < int(pool_counts[0]*args.initial_annotated_perc) or \
                count_p < int(pool_counts[1]*args.initial_annotated_perc)):
            if len(ind_list) >= sample_size:
                sample_size = sample_size
            else:
                sample_size = len(ind_list)
            inds = random.sample(ind_list, sample_size)
            for ind in inds:
                ind_list.remove(ind)
                if x_fp_pool[ind] != '-' and (count_np < int(pool_counts[0]*args.initial_annotated_perc) or \
                        count_p < int(pool_counts[1]*args.initial_annotated_perc)):
                    subdirs = x_fp_pool[ind].split('\\')
                    if subdirs[0] == CLASSES[0] and count_np < int(pool_counts[0]*args.initial_annotated_perc):
                        count_np += 1
                    elif subdirs[0] == CLASSES[1] and count_p < int(pool_counts[1]*args.initial_annotated_perc):
                        count_p += 1
                    else:
                        continue
                    x_fp_labeled.append(x_fp_pool[ind])
                    #print(count_np, count_p, ind, x_fp_pool[ind], len(x_fp_labeled))
                    
                    # get subdirectories before data file
                    if len(subdirs) <= 1:
                        temp_subdir = "" # subdirs[0]  
                    else:
                        temp_subdir = "\\".join(subdirs[:-1])            
                    
                    # move to corresponding folder
                    curr = TRAIN_LABELED_DIR 
                    if not os.path.exists(os.path.join(curr, temp_subdir)): # subdirs[0], subdirs[1]
                        os.makedirs(os.path.join(curr, temp_subdir)) # subdirs[0], subdirs[1]
                    shutil.move(os.path.join(TRAIN_DIR, x_fp_pool[ind]), 
                                os.path.join(curr, x_fp_pool[ind]))
                    
                    # "remove" from unlabeled pool (treat as null)
                    x_fp_pool[ind] = '-'    
       
    # update labeled and unlabeled data
    train_pool_gen, train_labeled_gen, x_fp_pool, x_fp_labeled, pool_counts = \
            refresh_unlabeled_labeled_generators(train_datagen, seed)
    print('Dataset initialised\n')
    return train_datagen, train_pool_gen, train_labeled_gen, val_generator, test_generator, \
            x_fp_pool, x_fp_labeled, len(CLASSES)
    
def create_resnet_model(n_classes):
    # load model without classifier layers
    model = ResNet50V2(include_top=False, weights='imagenet', input_tensor=None, 
                       input_shape=(DIM,DIM,3), #(DIM,DIM,3), #(32,32,3)
                       pooling=None, classes=n_classes) # pooling='avg' 
    
    if LAYER == 'GlobalAveragePool':
        x = GlobalAveragePooling2D(name="global_avg_pool")(model.layers[-1].output) #model.output #model.layers[-1].output
    elif LAYER == 'Flatten':
        x = Flatten()(model.layers[-1].output) # model.output #get_layer('top_activation').
    ##x = BatchNormalization()(x)
    
    # add new classifier layers
    x = Dense(1024, activation='relu', name="added_dense")(x)
    x = Dropout(0.5)(x)
    output = Dense(n_classes, activation='softmax', name="pred")(x)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    
    #model.summary()
    state = False
    for i, layer in enumerate(model.layers):
        # layers beyond stated layer are trainable
        if 'global_avg_pool' in layer.name:  #conv5_block3_1, global_avg_pool
            state = True
            #print(i, state)             #154, 190
        layer.trainable = state
        #print(layer.name, layer.trainable)
    #print(len(model.layers))    # 194
    #model.summary()
    return model

def create_effnetb7_model(n_classes):
    # load model without classifier layers
    model = EfficientNetB7(include_top=False, weights='imagenet', input_tensor=None, 
                           drop_connect_rate=0.2, # default 0.2
                           input_shape=(DIM,DIM,3), #(DIM,DIM,3), #(32,32,3)
                           pooling=None, classes=n_classes) # pooling='avg' #
    
    # Rebuild top
    if LAYER == 'GlobalAveragePool':
        x = GlobalAveragePooling2D(name="avg_pool")(model.layers[-1].output) #model.output
    elif LAYER == 'Flatten':
        x = Flatten()(model.layers[-1].output) # model.output #get_layer('top_activation').
    ##x = BatchNormalization()(x)
    
    # add new classifier layers
    x = Dense(1024, activation='relu')(x)
    top_dropout_rate = 0.5 #0.2
    x = Dropout(top_dropout_rate, name="top_dropout")(x)
    output = Dense(n_classes, activation='softmax', name="pred")(x)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    
    # model.summary()
    state = False
    for i, layer in enumerate(model.layers):
        # layers beyond stated layer are trainable
        if 'block6a_expand_conv' in layer.name:
            state = True
            #print(i)            # 558
        layer.trainable = state
        #print(layer.name, layer.trainable)
    #print(len(model.layers))    # 817
    #exit(0)
    return model
    
def create_effnetb0_model(n_classes):
    # load model without classifier layers
    model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=None, 
                           drop_connect_rate=0.2, # default 0.2
                           input_shape=(DIM,DIM,3), #(DIM,DIM,3), #(32,32,3)
                           pooling=None, classes=n_classes) # pooling='avg' #
    
    # Rebuild top
    if LAYER == 'GlobalAveragePool':
        x = GlobalAveragePooling2D(name="avg_pool")(model.layers[-1].output) #model.output
    elif LAYER == 'Flatten':
        x = Flatten()(model.layers[-1].output) # model.output #get_layer('top_activation').
    ##x = BatchNormalization()(x)
    
    # add new classifier layers
    x = Dense(512, activation='relu')(x)
    top_dropout_rate = 0.5 #0.2
    x = Dropout(top_dropout_rate, name="top_dropout")(x)
    output = Dense(n_classes, activation='softmax', name="pred")(x)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    
    # model.summary()
    state = False
    for i, layer in enumerate(model.layers):
        # layers beyond stated layer are trainable
        if 'block6a_expand_conv' in layer.name:
            state = True
            # print(i)            #162
        layer.trainable = state
        #print(layer.name, layer.trainable)
    # print(len(model.layers))    #241
    return model

def create_effnetb6_model(n_classes):
    # load model without classifier layers
    model = EfficientNetB6(include_top=False, weights='imagenet', input_tensor=None, 
                           # drop_connect_rate=0.2, # default 0.2
                           input_shape=(DIM,DIM,3), #(DIM,DIM,3), #(32,32,3)
                           pooling=None, classes=n_classes) # pooling='avg' #
    
    # Rebuild top
    if LAYER == 'GlobalAveragePool':
        x = GlobalAveragePooling2D(name="avg_pool")(model.layers[-1].output) #model.output
    elif LAYER == 'Flatten':
        x = Flatten()(model.layers[-1].output) # model.output #get_layer('top_activation').
    ##x = BatchNormalization()(x)
    
    # add new classifier layers
    x = Dense(1024, activation='relu')(x)
    top_dropout_rate = 0.5 #0.2
    x = Dropout(top_dropout_rate, name="top_dropout")(x)
    output = Dense(n_classes, activation='softmax', name="pred")(x)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    
    # model.summary()
    state = False
    for i, layer in enumerate(model.layers):
        # layers beyond stated layer are trainable
        if 'block7a_expand_conv' in layer.name:
            state = True
            #print(i)            # block6: 456, block7: 620
        layer.trainable = state
        #print(layer.name, layer.trainable)
    #print(len(model.layers))    # 670
    #exit(0)
    return model
    
def initialize_model(train_datagen, train_pool_gen, train_labeled_gen, val_generator, test_generator, n_classes):
    if RESET:
        if os.path.exists(args.chkt_filename_init):
            print(f'Loading {args.chkt_filename_init}...')
            model = load_model(args.chkt_filename_init)
        else:
            if CNN_MODEL == CNN[0]:  # ResNet50_V2
                model = create_resnet_model(n_classes)
            elif CNN_MODEL == CNN[1]:    #EffNet-B7
                model = create_effnetb7_model(n_classes)
            elif CNN_MODEL == CNN[2]:    #EffNet-B0
                model = create_effnetb0_model(n_classes)
            elif CNN_MODEL == CNN[3]:    #EffNet-B6
                model = create_effnetb6_model(n_classes)
                
            #opt = Adam(learning_rate=0.001)
            model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['acc'])
            # keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5),
            # keras.metrics.FalseNegatives(),
            # keras.metrics.FalsePositives(),
        
            if args.verbose != 1:
                print('Training initial model')
            step_size_train = math.ceil(train_labeled_gen.n/train_labeled_gen.batch_size) #train_labeled_gen.n // train_labeled_gen.batch_size
            step_size_valid = math.ceil(val_generator.n/val_generator.batch_size)#test_generator.n // test_generator.batch_size
            history = model.fit(train_labeled_gen, steps_per_epoch=step_size_train,
                                validation_data=val_generator, validation_steps=step_size_valid, #validation_data=test_generator
                                epochs=args.epochs, verbose=args.verbose, callbacks=[checkpoint_init, earlystop_init], 
                                use_multiprocessing=args.multiprocessing, workers=args.workers) 
            
            ind_best = history.history['val_loss'].index(min(history.history['val_loss']))
            best_val_loss, best_val_acc = history.history['val_loss'][ind_best], history.history['val_acc'][ind_best]*100 
            best_train_loss, best_train_acc = history.history['loss'][ind_best], history.history['acc'][ind_best]*100
            
            plot_train_val_graph(train_hist=history.history['acc'], val_hist=history.history['val_acc'], 
                                 title=f'Initial Model Accuracy (Best:{best_train_acc:.2f}; {best_val_acc:.2f})', 
                                 ylabel='Accuracy', legend_loc='lower right',
                                 savename='init_acc', show_image=False, savefig=True)
            plot_train_val_graph(train_hist=history.history['loss'], val_hist=history.history['val_loss'], 
                                 title=f'Initial Model Loss (Best:{best_train_loss:.4f}; {best_val_loss:.4f})', 
                                 ylabel='Loss', legend_loc='upper right',
                                 savename='init_loss', show_image=False, savefig=True)
            
            #for (v_acc, v_loss) in zip(history.history['val_acc'], history.history['val_loss']):
            
            # load best initial model
            model = load_model(args.chkt_filename_init)
    else:
        if os.path.exists(args.chkt_filename):
            print(f'Loading {args.chkt_filename}...')
            model = load_model(args.chkt_filename)
        else:
            print(f'Error in loading {args.chkt_filename}')
            exit(0)
            
    scores = model.evaluate(test_generator, verbose=args.verbose) #steps=step_size_valid, 
    print('Initial: Test Loss {0:.4f}  Test Accuracy: {1:.3f}'.format(scores[0], scores[1]*100))
    
    #step_size_test = test_generator.n//test_generator.batch_size
    pred = np.argmax(model.predict(test_generator, verbose=args.verbose),axis=1) #steps=step_size_test, 
        
    classification_report_csv(y_true=test_generator.labels, y_pred=pred, digits=4, filename='init_report.csv', display=True)
    cnf_matrix = confusion_matrix(y_true=test_generator.labels, y_pred=pred, labels=None, normalize=None)
    plot_confusion_matrix(cnf_matrix, classes=None, n_classes=n_classes,
                          title=f'Confusion matrix_0.0', savename=f'ConfusionMatrix_0.0.png',
                          savefig=True)
    #exit(0)
    return model, scores


''' Selection related '''
# Random sampling
def random_sampling(y_pred_prob, n_samples): # added replace
    if len(y_pred_prob) < n_samples:
        n_samples = len(y_pred_prob)
        print(f'\tsample size: {len(y_pred_prob)}')
    return np.random.choice(range(len(y_pred_prob)), n_samples, replace=False)

# Rank all the unlabeled samples in an ascending order according to the least confidence
def least_confidence(y_pred_prob, n_samples):
    origin_index = np.arange(0, len(y_pred_prob))
    max_prob = np.max(y_pred_prob, axis=1)
    pred_label = np.argmax(y_pred_prob, axis=1)

    lci = np.column_stack((origin_index,
                           max_prob,
                           pred_label))
    lci = lci[lci[:, 1].argsort()]
    return lci[:n_samples], lci[:, 0].astype(int)[:n_samples]

# Rank all the unlabeled samples in an ascending order according to the margin sampling
def margin_sampling(y_pred_prob, n_samples):
    origin_index = np.arange(0, len(y_pred_prob))
    margim_sampling = np.diff(-np.sort(y_pred_prob)[:, ::-1][:, :2])
    pred_label = np.argmax(y_pred_prob, axis=1)
    msi = np.column_stack((origin_index,
                           margim_sampling,
                           pred_label))
    msi = msi[msi[:, 1].argsort()]
    return msi[:n_samples], msi[:, 0].astype(int)[:n_samples]

# Rank all the unlabeled samples in an descending order according to their entropy
def entropy(y_pred_prob, n_samples):
    # entropy = stats.entropy(y_pred_prob.T)
    # entropy = np.nan_to_num(entropy)
    origin_index = np.arange(0, len(y_pred_prob))
    entropy = -np.nansum(np.multiply(y_pred_prob, np.log(y_pred_prob)), axis=1)
    pred_label = np.argmax(y_pred_prob, axis=1)
    eni = np.column_stack((origin_index,
                           entropy,
                           pred_label))

    eni = eni[(-eni[:, 1]).argsort()]   
    #print(eni, type(eni), eni.shape)
    return eni[:n_samples], eni[:, 0].astype(int)[:n_samples]

def get_high_confidence_samples(y_pred_prob, delta):
    eni, eni_idx = entropy(y_pred_prob, len(y_pred_prob))
    hcs = eni[eni[:, 1] < delta]    # select low entropy data
    return hcs[:, 0].astype(int), hcs[:, 2].astype(int) # data indices, data labels

def get_uncertain_samples(y_pred_prob, n_samples, criteria):
    if criteria == 'lc':
        return least_confidence(y_pred_prob, n_samples)
    elif criteria == 'ms':
        return margin_sampling(y_pred_prob, n_samples)
    elif criteria == 'en':
        return entropy(y_pred_prob, n_samples)
    elif criteria == 'rs':
        return None, random_sampling(y_pred_prob, n_samples)
    else:
        raise ValueError(
            'Unknown criteria value \'%s\', use one of [\'rs\',\'lc\',\'ms\',\'en\']' % criteria)


''' Refresh generators and filename lists'''
def refresh_unlabeled_labeled_generators(train_datagen, seed, init=False):
    # update labeled and unlabeled data
    train_pool_gen = train_datagen.flow_from_directory(TRAIN_DIR,
                                                        target_size=(DIM,DIM),  # resize
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        class_mode='categorical',   # e.g., [0. 1.]
                                                        color_mode='rgb',
                                                        seed=seed)
    train_labeled_gen = train_datagen.flow_from_directory(TRAIN_LABELED_DIR,
                                                            target_size=(DIM,DIM),  # resize
                                                            batch_size=args.batch_size,
                                                            shuffle=True,
                                                            class_mode='categorical',   # e.g., [0. 1.]
                                                            color_mode='rgb',
                                                            seed=seed)
    # print('UPDATED: ', train_pool_gen.samples, train_labeled_gen.samples)
    
    # lists of training (labeled and unlabeled) image filepaths
    x_fp_pool, x_fp_labeled = train_pool_gen.filenames, train_labeled_gen.filenames
    if init:
        x_fp_labeled = []
        for fp in train_labeled_gen.filenames:
            if 'hc' in fp.split("\\"):
                continue
            else:
                x_fp_labeled.append(fp)
    else:
        x_fp_labeled = train_labeled_gen.filenames
    
    _, pool_counts = np.unique(train_pool_gen.labels, return_counts=True)
    #print(pool_counts)
    return train_pool_gen, train_labeled_gen, x_fp_pool, x_fp_labeled, pool_counts
   
   
''' Movement of data (labeled, unlabeled, high confidence) '''
def move_unlabeled_to_labeled(train_pool_gen, train_labeled_gen, un_idx, x_fp_pool, x_fp_labeled,
                              train_datagen, seed):
    # move selected unlabeled data to labeled
    count_np, count_p = 0, 0
    for ind in un_idx:
        if ind >= len(x_fp_pool):
            print(ind, len(x_fp_pool))
            exit(0)
        subdirs = x_fp_pool[ind].split('\\')
        if subdirs[0] == CLASSES[0]:
            count_np += 1
        elif subdirs[0] == CLASSES[1]:
            count_p += 1
        x_fp_labeled.append(x_fp_pool[ind])
        #print(count_np, count_p, ind, x_fp_pool[ind], len(x_fp_labeled))
        
        # get subdirectories before data file
        if len(subdirs) <= 1:
            temp_subdir = "" # subdirs[0]  
        else:
            temp_subdir = "\\".join(subdirs[:-1])        
        #print(temp_subdir)
        
        # move to corresponding folder
        if not os.path.exists(os.path.join(TRAIN_LABELED_DIR, temp_subdir)): #subdirs[0], subdirs[1]
            os.makedirs(os.path.join(TRAIN_LABELED_DIR, temp_subdir))
        shutil.move(os.path.join(TRAIN_DIR, x_fp_pool[ind]), 
                    os.path.join(TRAIN_LABELED_DIR, x_fp_pool[ind]))
        
    # update labeled and unlabeled data
    train_pool_gen, train_labeled_gen, x_fp_pool, x_fp_labeled, pool_counts = \
            refresh_unlabeled_labeled_generators(train_datagen, seed)
    return train_pool_gen, train_labeled_gen, x_fp_pool, x_fp_labeled

def move_unlabeled_to_high_confidence(train_pool_gen, hc_idx, hc_labels, 
                                      x_fp_pool, train_datagen, seed,
                                      train_labeled_gen, x_fp_labeled):
    # move selected unlabeled data to high confidence
    count_np, count_p = 0, 0
    hc_incorrect, pred_labels = [], []  #true_labels
    for (ind, label) in zip(hc_idx, hc_labels):
        subdirs = x_fp_pool[ind].split('\\')
        if subdirs[0] == CLASSES[0]:
            count_np += 1
        elif subdirs[0] == CLASSES[1]:
            count_p += 1
        x_fp_labeled.append(x_fp_pool[ind])  # x_fp_hc
        #print(count_np, count_p, ind, x_fp_pool[ind], len(x_fp_hc))
        
        # Output incorrect labels
        if CLASSES[label] != subdirs[0]:
            pred_labels.append(label)
            hc_incorrect.append(x_fp_pool[ind])
        
        # get subdirectories before data file
        if len(subdirs) <= 2:
            temp_subdir = ""     
        else:
            temp_subdir = "\\".join(subdirs[1:-1])       
        #print(subdirs, temp_subdir, subdirs[-1])
        # Copy based on predicted label
        if not os.path.exists(os.path.join(TRAIN_LABELED_DIR, CLASSES[label], 'hc', temp_subdir)): #subdirs[1]
            os.makedirs(os.path.join(TRAIN_LABELED_DIR, CLASSES[label], 'hc', temp_subdir))
        shutil.copy(os.path.join(TRAIN_DIR, x_fp_pool[ind]), 
                    os.path.join(TRAIN_LABELED_DIR, CLASSES[label], 'hc', temp_subdir, subdirs[-1])) 
        
        # get subdirectories before data file
        if len(subdirs) <= 1:
            temp_subdir = "" # subdirs[0]  
        else:
            temp_subdir = "\\".join(subdirs[:-1]) 
        #print(subdirs, temp_subdir)
        # Move according to true label
        if not os.path.exists(os.path.join(TRAIN_HC_DIR, temp_subdir)): #subdirs[0], subdirs[1]
            os.makedirs(os.path.join(TRAIN_HC_DIR, temp_subdir))
        shutil.move(os.path.join(TRAIN_DIR, x_fp_pool[ind]), 
                    os.path.join(TRAIN_HC_DIR, x_fp_pool[ind]))
    
        
    # update labeled and unlabeled data
    train_pool_gen, train_labeled_gen, x_fp_pool, x_fp_labeled, pool_counts = \
            refresh_unlabeled_labeled_generators(train_datagen, seed)
    return train_pool_gen, train_labeled_gen, x_fp_pool, x_fp_labeled, hc_incorrect, pred_labels 

def return_high_confidence_to_unlabeled(x_fp_labeled):
    # move to unlabeled
    for c in CLASSES:
        if os.path.exists(os.path.join(TRAIN_HC_DIR, c)):
            for var in os.listdir(os.path.join(TRAIN_HC_DIR, c)):
                if os.path.isdir(os.path.join(TRAIN_HC_DIR, c, var)):
                    for fp in os.listdir(os.path.join(TRAIN_HC_DIR, c, var)):
                        shutil.move(os.path.join(TRAIN_HC_DIR, c, var, fp), 
                                    os.path.join(TRAIN_DIR, c, var, fp))
                else:
                    shutil.move(os.path.join(TRAIN_HC_DIR, c, var), 
                                os.path.join(TRAIN_DIR, c, var))    

    # remove hc contents in labeled
    for fp in x_fp_labeled:
        if 'hc' in fp.split("\\"):
            os.remove(os.path.join(TRAIN_LABELED_DIR, fp))
    # check if deleted files
    try:
        for c in CLASSES:
            if len(os.listdir(os.path.join(TRAIN_LABELED_DIR, c, 'hc', 'true'))) != 0:
                print('HC files not deleted from labeled folder!')
                exit(0)
    except Exception as e:
        print(f"{os.path.join(TRAIN_LABELED_DIR, c, 'hc', 'true')} does not exist.")
   
''' CEAL Algorithm '''    
def run_ceal(args):
    # # check if WEBP support installed in Pillow
    # print (features.check_module('webp'))
    # exit(0)
    
    create_folders()
    
    if not RESET_ONLY:
        # extract images from class folders (if applicable)
        extract_images_from_subfolders()
        
        # split training and validation set
        manual_split_train_val_dataset(seed=args.seed, val_split=args.val_split)
    # else:
        # if 'extended' in TESTFOLD: # prepare test set
            # move_fold_test_data(filename=args.extended_fold_textfile)
            
    # prepare initial labeled dataset for model training
    train_datagen, train_pool_gen, train_labeled_gen, val_generator, test_generator, x_fp_pool, x_fp_labeled, \
        n_classes = initialize_custom_dataset(seed=args.seed, sample_size=500)
    # initial total
    total_training_data = len(x_fp_pool) + train_labeled_gen.samples
    perc = 0.2
    print('TOTAL: ', total_training_data)
    # # define max number of labeled data; if exceed this value then stop AL iteration
    # max_labeled = int((len(x_fp_pool) + train_labeled_gen.samples) * args.max_labeled)
    # print('MAX labeled: ', max_labeled)
    
    # initialise model
    perc_x, perc_time, perc_test_acc, cnf_matrices = [], [], [], [] #int(args.initial_annotated_perc*100)
    model, scores = initialize_model(train_datagen, train_pool_gen, train_labeled_gen, val_generator,
                                     test_generator, n_classes)
    tloss_history, taccuracy_history = [scores[0]], [scores[1]]
    best_model, best_loss, best_acc = model, scores[0], scores[1]
    #perc_test_acc.append(scores[1]) # initial labeled test accuracy
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    #perc_time.append(total_time_str)
    
    w, h, c = DIM, DIM, 3  
    #print(train_pool_gen.target_size) #(224,224)
    
    # unlabeled samples # train_pool_gen WITHOUT shuffling
    DU = train_datagen.flow_from_directory(TRAIN_DIR,
                                            target_size=(DIM,DIM),  # resize
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            class_mode='categorical',   # e.g., [0. 1.]
                                            color_mode='rgb',
                                            seed=args.seed)
    
    # initially labeled samples #train_labeled_gen WITH shuffling
    DL = train_labeled_gen
    
    #print(len(DU))  # samples/batch size
    
    end_time_list, pred_labels_list, hc_record, labeled_record = [], [], [], [] 
    pred_labels_np, pred_labels_p = [], []
    tacc_list, vacc_list, tloss_list, vloss_list, epoch_list = [], [], [], [], []
    hc, pred_labels, hc_incorrect = [], [], []
    # try:
    for i in range(args.maximum_iterations):
        hc_incorrect_list, pred_labels_list = [], []
        flag = True
        
        print(f'refreshed: {int(perc*100)}% is {int(total_training_data * perc)}, {DL.samples-len(hc)} labeled')
        model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['acc'])
         
        # if DL.samples >= max_labeled:
            # print(f'Enough labeled data ({int(args.max_labeled*100)}%)! [{DL.samples}>={max_labeled}]')
            # break
        while flag:
            #if (DL.samples-len(hc)) >= int(total_training_data * perc):
            
            perc = (DL.samples-len(hc))/total_training_data
            #print(f'Enough labeled data ({int(perc*100)}%)! [{DL.samples-len(hc)}>={int(total_training_data * perc)}]')
            test_acc, cnf_matrix = eval_best_model(test_generator=test_generator, model=model, perc=perc, n_classes=n_classes, args=args)
            perc_test_acc.append(test_acc)
            cnf_matrices.append(f'[{cnf_matrix[0][0]}, {cnf_matrix[0][1]}, {cnf_matrix[1][0]}, {cnf_matrix[1][1]}]') #cnf_matrix
            perc_x.append(perc*100) #int(perc*100)
            #perc += 0.2
            
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            perc_time.append(total_time_str)
        
            for (x, acc, t) in zip(perc_x, perc_test_acc, perc_time):
                print(f'[{x}, {acc*100:.2f}, {t}] \t')
            print('\n')
            
            if perc >= 1.0:
                break
            # else:
            flag = False
                
        # update labeled and unlabeled data in DirectoryIterator
        DU, DL, x_fp_DU, x_fp_DL, pool_counts = \
                refresh_unlabeled_labeled_generators(train_datagen, args.seed)
        
        # finalise if all data labeled 
        if DU.samples == 0:
            print('All labeled')
            break
            
        y_pred_prob = model.predict(DU, verbose=args.verbose) #steps=DU.n//DU.batch_size
        #print(y_pred_prob.shape) #45000,10
        
        _, un_idx = get_uncertain_samples(y_pred_prob, args.uncertain_samples_size, criteria=args.uncertain_criteria)
        #print(un_idx, len(un_idx))
        # # output number of unique elements in list (if replacement allowed for random sampling)
        # seen = set()
        # uniq = [x for x in un_idx if x in seen or seen.add(x)]    
        # print(len(uniq))
        
        # for j in un_idx:  #list out of index
            # print(CLASSES.index(x_fp_DU[j].split("\\",1)[0]), x_fp_DU[j], y_pred_prob[j])
         
        DU, DL, x_fp_DU, x_fp_DL = move_unlabeled_to_labeled(train_pool_gen=DU, 
                                                            train_labeled_gen=DL, 
                                                            un_idx=un_idx, 
                                                            x_fp_pool=x_fp_DU, 
                                                            x_fp_labeled=x_fp_DL, 
                                                            train_datagen=train_datagen, 
                                                            seed=args.seed)
        print("\nLabeled data")
        
        # print('here: ', max(un_idx), len(y_pred_prob))  #170, 177
        un_idx = np.sort(un_idx)[::-1]  # descending order
        # print(un_idx, un_idx.shape)
        
        for ind in un_idx:
            y_pred_prob = np.delete(y_pred_prob, ind, axis=0)
        #print(y_pred_prob.shape)
        
        if args.cost_effective:
            hc_idx, hc_labels = get_high_confidence_samples(y_pred_prob, args.delta)
            ## set max
            
                
            # remove samples also selected through uncertain (duplicates)
            hc = np.array([[i, l] for i, l in zip(hc_idx, hc_labels) if i not in un_idx])
            
            # keep track of numbers
            hc_record.append(len(hc))
            labeled_record.append(DL.samples)
            
            print("HC     : "+ "\t".join([str(x) for x in hc_record]))
            print("labeled: " + "\t".join([str(x) for x in labeled_record]))

            if hc.size != 0: 
                print('\n')
                # prepare generator of labeled data for training (with shuffle)
                DU, DL, x_fp_DU, x_fp_DL, hc_incorrect, pred_labels  = \
                            move_unlabeled_to_high_confidence(train_pool_gen=DU,  
                                                               hc_idx=hc[:, 0], 
                                                               hc_labels=hc[:, 1], 
                                                               x_fp_pool=x_fp_DU, 
                                                               train_datagen=train_datagen, 
                                                               seed=args.seed,
                                                               train_labeled_gen=DL, #train_labeled_gen, 
                                                               x_fp_labeled=x_fp_DL) #x_fp_labeled)
                #hc_incorrect_list = hc_incorrect_list + hc_incorrect
                #pred_labels_list = pred_labels_list + pred_labels 
                pred_labels_np.append(pred_labels.count(0))
                pred_labels_p.append(pred_labels.count(1))
            else:
                print('No HC')
                pred_labels_np.append(0)
                pred_labels_p.append(0)
            print(f'DL+HC: {DL.samples}[{DL.samples-len(hc)}+{len(hc)}]   DU:{DU.samples} ' + \
                        f'[{pred_labels.count(0)}+{pred_labels.count(1)} errors]')
            
        else:
            labeled_record.append(DL.samples)
            
        if i % args.fine_tunning_interval == 0:     
            history = []
            step_size_train = math.ceil(DL.n/DL.batch_size) #DL.n//DL.batch_size
            step_size_valid = math.ceil(val_generator.n/val_generator.batch_size) #test_generator.n//test_generator.batch_size
            history = model.fit(DL, validation_data=val_generator, #batch_size=args.batch_size,
                                shuffle=True, epochs=args.epochs, verbose=args.verbose, 
                                callbacks=[earlystop, checkpoint], steps_per_epoch=step_size_train, #lr_reducer
                                validation_steps=step_size_valid,
                                use_multiprocessing=args.multiprocessing, workers=args.workers)
            #if i == 0:
            # plot_train_val_graph(train_hist=history.history['acc'], val_hist=history.history['val_acc'], 
                                 # title=f'Model Accuracy (Iter{i})', ylabel='Accuracy', legend_loc='lower right',
                                 # savename=f'later_acc_{i}', show_image=False, savefig=True)
            # plot_train_val_graph(train_hist=history.history['loss'], val_hist=history.history['val_loss'], 
                                 # title=f'Model Loss (Iter{i})', ylabel='Loss', legend_loc='upper right',
                                 # savename=f'later_loss_{i}', show_image=False, savefig=True)
                  
            args.delta -= (args.threshold_decay * (i + 1))   # for entropy calculation
            #args.delta -= (args.threshold_decay * args.fine_tunning_interval)   # for entropy calculation
        
            val_loss_min = np.min(history.history['val_loss'])
            epoch_min = np.argmin(history.history['val_loss'])
            val_acc_at_epoch_min = history.history['val_acc'][epoch_min]
            train_acc_at_epoch_min = history.history['acc'][epoch_min]
            train_loss_at_epoch_min = history.history['loss'][epoch_min]
            #print(epoch_min, val_loss_min, val_acc_at_epoch_min, train_acc_at_epoch_min, train_loss_at_epoch_min)
            tacc_list.append(train_acc_at_epoch_min)
            vacc_list.append(val_acc_at_epoch_min)
            tloss_list.append(train_loss_at_epoch_min)
            vloss_list.append(val_loss_min)
            epoch_list.append(epoch_min)
            
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            end_time_list.append(total_time_str)
            
        loss, acc  = model.evaluate(test_generator, verbose=args.verbose) #steps=step_size_valid, 
        tloss_history.append(loss)
        taccuracy_history.append(acc)
        print(
            'Iteration: %d; [HC: %d; Uncertain: %d; Delta: %.5f; Labeled: %d]; Accuracy: %.2f; Loss: %.3f'
            % (i, len(hc), len(un_idx), args.delta, DL.samples-len(hc), acc*100, loss))
        # print(
            # 'Iteration: %d; High Confidence Samples: %d; Uncertain Samples: %d; Delta: %.5f; Labeled Dataset Size: %d; Accuracy: %.2f'
            # % (i, len(DH[0]), len(DL[0]), args.delta, len(DL[0]), acc))
    
        if args.cost_effective:
            # Erase pseudolabels
            return_high_confidence_to_unlabeled(x_fp_labeled=x_fp_DL)
        
        # save model
        # if loss <= tloss_history[-1]:   # test loss
            # model.save(f"my_resnet50v2_custom{TESTFOLD}_model.hdf5")
            # best_model, best_loss, best_acc = model, loss, acc
            # print('Best model updated')
        # else:
            # print('No improvement')
        # # train with different HC data
        # best_model = model
    
        if args.cost_effective:
            #print(len(hc_incorrect), len(pred_labels))
            # write wrongly predicted high confidence data to textfile
            write_wrong_hc_to_textfile(hc_list=hc_incorrect, pred_labels=pred_labels, 
                                       filename=f'wrong_hc_{i}.txt')
    
    # print('Best: Test Loss {0:.4f}  Test Accuracy {1:.3f}'.format(best_loss, best_acc*100))
    
    return tloss_history, taccuracy_history, n_classes, hc_record, labeled_record, \
            pred_labels_np, pred_labels_p, history, tacc_list, vacc_list, tloss_list, vloss_list, \
            epoch_list, end_time_list, perc_x, perc_time, perc_test_acc, cnf_matrices #cnf_matrix
    # except Exception as e:
        # print(e)

def eval_best_model(test_generator, model, perc, n_classes, args):
    # Load best model
    try:
        best_model = load_model(args.chkt_filename)
    except:
        best_model = model 
    loss, acc  = best_model.evaluate(test_generator, verbose=args.verbose)
    
    #step_size_test = test_generator.n//test_generator.batch_size
    pred = np.argmax(best_model.predict(test_generator, verbose=args.verbose),axis=1) #steps=step_size_test, 
    classification_report_csv(y_true=test_generator.labels, y_pred=pred, digits=4, filename=f'report_{perc:.1f}.csv', display=False)
    cnf_matrix = confusion_matrix(y_true=test_generator.labels, y_pred=pred, labels=None, normalize=None)
    plot_confusion_matrix(cnf_matrix, classes=None, n_classes=n_classes,
                          title=f'Confusion matrix_{perc:.1f}', savename=f'ConfusionMatrix_{perc:.1f}.png',
                          savefig=True)
    return acc, cnf_matrix    
    
    
    


''' Plotting graphs and exporting report '''
def plot_graph(performance_history, xvals, title, ylabel, xlabel, savename, savefig=True):
    # Plot our performance over time.
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)

    ax.plot(xvals, performance_history)
    ax.scatter(xvals, performance_history, s=13)

    #ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=args.maximum_iterations + 3, integer=True))
    ax.xaxis.grid(True)
    ax.set_xlim(left=min(xvals), right=max(xvals))
    
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    #ax.set_ylim(bottom=0, top=1)
    ax.yaxis.grid(True, linestyle='--', alpha=1/2)

    ax.set_title(f'{title} (Average: {np.mean(performance_history)*100:.3f}, ' + \
                 f'Best:{np.max(performance_history)*100:.3f})')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if savefig:
        plt.savefig(f'{savename}.png', transparent=True, dpi=100)
    plt.show()
    plt.close(fig) 
    
# https://datascience.stackexchange.com/questions/40067/confusion-matrix-three-classes-python
def plot_confusion_matrix(cm,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          classes=None,
                          n_classes=2,
                          savename='Confusion matrix.png',
                          savefig=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    figx = plt.figure()
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if classes == None:
        classes = [i for i in range(n_classes)]
        #print(classes)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()   

    if savefig:
        plt.savefig(savename, transparent=True, dpi=100)
    #plt.show()
    plt.close(figx) 

def plot_train_val_graph(train_hist, val_hist, title, ylabel, legend_loc, savename, 
                         show_image=True, savefig=True):
    fig = plt.figure()
    xc = range(len(train_hist))
    plt.plot(xc, train_hist)
    plt.plot(xc, val_hist)
    
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc=legend_loc)
    
    if savefig: 
        print('saving image')
        plt.savefig(f'{savename}.png', transparent=True, dpi=100)
    if show_image:
        plt.show()
    
    plt.close(fig) 
    

def classification_report_csv(y_true, y_pred, filename, digits=4, display=True, output_dict=True, export=True):
    if display:
        report = classification_report(y_true=y_true, y_pred=y_pred, digits=4)
        print(report)
    if export:
        dataframe = pd.DataFrame(classification_report(y_true=y_true, y_pred=y_pred, digits=4, output_dict=True)).transpose()
        dataframe.to_csv(filename, index=True)

''' Prepare folders and write textfile'''
def create_folders():
    try:
        curr = os.path.join(TRAIN_HC_DIR) 
        if not os.path.exists(curr):
            os.makedirs(curr)    
        
        for c in CLASSES:
            if not os.path.exists(os.path.join(curr, c)):
                os.makedirs(os.path.join(curr, c))
    except Exception as e:
        print(f'Error in creating hc folder branch: {e}')
        exit(0)
    
    try:
        curr = os.path.join(TRAIN_LABELED_DIR) 
        if not os.path.exists(curr):
            os.makedirs(curr)
        for c in CLASSES:
            if not os.path.exists(os.path.join(curr, c)):
                os.makedirs(os.path.join(curr, c))
            
            if not os.path.exists(os.path.join(curr, c, 'hc')):
                os.makedirs(os.path.join(curr, c, 'hc'))
    except Exception as e:
        print(f'Error in creating labeled folder branch: {e}')   
        exit(0)
    
    # val/<class names>
    try:
        curr = os.path.join(TRAIN_VAL_DIR) 
        if not os.path.exists(curr):
            os.makedirs(curr)    
        
        for c in CLASSES:
            if not os.path.exists(os.path.join(curr, c)):
                os.makedirs(os.path.join(curr, c))
            
            if not os.path.exists(os.path.join(curr, c, 'true')):
                os.makedirs(os.path.join(curr, c, 'true'))
    except Exception as e:
        print(f'Error in creating val folder branch: {e}')
        exit(0)
    
    print('Folders created\n')

def write_val_data_to_textfile(val_list, filename):
    if len(val_list) == 0:
        print('No val data')
        return
        
    with open(filename, 'w') as f:
        for val_data in val_list:
            try:
                f.write(f'{val_data}\n')
            except Exception as e:
                print(val_data)
    print(f'Written to {filename}' + '\n')
    
def write_wrong_hc_to_textfile(hc_list, pred_labels, filename):
    if len(hc_list) == 0:
        print('No hc')
        return
        
    with open(filename, 'w') as f:
        for (label, im) in zip(pred_labels, hc_list):
            try:
                f.write(f'{label}\t {im}\n')
            except Exception as e:
                print(im, e)
    print(f'Written to {filename}' + '\n')

def write_to_textfile_finalise(train_time, hc_np, hc_p, hc_total, labeled_total, accuracy_history, history, \
                                tacc_list, vacc_list, tloss_list, vloss_list, epoch_list, end_time_list, \
                                perc_x, perc_time, perc_test_acc, cnf_matrices, filename):
    with open(filename, 'w') as f:
        f.write('Training time {}\n\n'.format(total_time_str))
        f.write('wrong_HC_0          : ' + "\t".join([str(x) for x in hc_np]))
        f.write('\n')
        f.write('wrong_HC_1          : ' + "\t".join([str(x) for x in hc_p]))
        f.write('\n')
        f.write('HC data count       : ' + "\t".join([str(x) for x in hc_total]))
        f.write('\n')
        f.write('Labeled data count  : ' + "\t".join([str(x) for x in labeled_total])) 
        f.write('\nTest Accuracy       : ' + "\t".join(["%.2f "%(x*100) for x in accuracy_history]))         
        f.write('\nAverage accuracy    : {avgacc:.3f}'.format(avgacc=np.mean(accuracy_history)*100))
        
        f.write('\n')
        f.write('\nEpoch              : \t' + "\t".join([str(x) for x in epoch_list]))
        f.write('\nTrain accuracy     : \t' + "\t".join(["%.2f "%(x*100) for x in tacc_list]))         
        f.write('\nVal accuracy       : \t' + "\t".join(["%.2f "%(x*100) for x in vacc_list]))         
        f.write('\nTrain loss         : \t' + "\t".join(["%.3f "%x for x in tloss_list]))         
        f.write('\nVal loss           : \t' + "\t".join(["%.3f "%x for x in vloss_list]))         
        f.write('\nAverages           : \t{tacc:.3f} \t{vacc:.3f} \t{tloss:.4f} \t{vloss:.4f}'.format(\
                    tacc=np.mean(tacc_list)*100, vacc=np.mean(vacc_list)*100, \
                    tloss=np.mean(tloss_list), vloss=np.mean(vloss_list)))
        f.write('\n\nEnd time         : \t' + "\t".join([str(x) for x in end_time_list]))  
        
        f.write('\n')
        f.write('\nPercentage         : \t' + ", ".join(["%.2f"%(x) for x in perc_x]))
        f.write('\nTest Accuracy      : \t' + ", ".join(["%.2f"%(x*100) for x in perc_test_acc]))
        f.write('\nTime               : \t' + ", ".join([str(x) for x in perc_time]))  
        
        f.write('\n')
        f.write('\nConfusion Matrix   : \t' + ", ".join([str(x) for x in cnf_matrices]))  
    print(f'Written to {filename}' + '\n')
       
    

''' Main function '''
if __name__ == '__main__':
    gc.collect()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-verbose', default=1, type=int,
                        help="Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. default: 0")
    parser.add_argument('-epochs', default=5, type=int, help="Number of epoch to train. default: 5") #30
    parser.add_argument('-batch_size', default=32, type=int, 
                        help="Number of samples per gradient update. default: 32")
    parser.add_argument('-chkt_filename_init', default=f"chkpt_{CNN_MODEL}_fold{TESTFOLD}_init.hdf5",  #"ResNet18v2-CIFAR-10_init_ceal.hdf5"
                        help="Model Initial Checkpoint filename to save")
    parser.add_argument('-chkt_filename', default=f"chkpt_{CNN_MODEL}_fold{TESTFOLD}.hdf5",  #"ResNet18v2-CIFAR-10_init_ceal.hdf5"
                        help="Model Checkpoint filename to save")
    parser.add_argument('-t', '--fine_tunning_interval', default=1, type=int, 
                        help="Fine-tuning interval. default: 1")
    parser.add_argument('-T', '--maximum_iterations', default=50, type=int, #AL iteration; 45
                        help="Maximum iteration number. default: 10")
    parser.add_argument('-i', '--initial_annotated_perc', default=0.2, type=float, #0.2 #0.1
                        help="Initial Annotated Samples Percentage. default: 0.1")
    parser.add_argument('-dr', '--threshold_decay', default=0.0033, type=float,
                        help="Threshold decay rate. default: 0.0033")
    parser.add_argument('-delta', default=0.05, type=float,
                        help="High confidence samples selection threshold. default: 0.05")
    parser.add_argument('-K', '--uncertain_samples_size', default=500, type=int, #1000 #30 #150/200/250 #50
                        help="Uncertain samples selection size. default: 1000")
    # parser.add_argument('-max_labeled', default=0.2, type=float,
                        # help="Stops iteration if labeled samples size exceeds this percentage. default: 0.5")
    parser.add_argument('-uc', '--uncertain_criteria', default='lc',
                        help="Uncertain selection Criteria: \'rs\'(Random Sampling), \'lc\'(Least Confidence)," + 
                                " \'ms\'(Margin Sampling), \'en\'(Entropy). default: lc")
    parser.add_argument('-ce', '--cost_effective', default=False,
                        help="whether to use Cost Effective high confidence sample pseudo-labeling. default: True")
    parser.add_argument('-seed', default=33, type=int, #42, 14, 27, 59, 33
                        help="Allows result to be reproduced. default: 1")
    parser.add_argument('-multiprocessing', default=False, type=bool,
                        help="Whether to use process-based threading. default: True")
    parser.add_argument('-workers', default=1, type=int, 
                        help="Maximum number of processes when using process-based threading. default: 1")
    parser.add_argument('-val_split', default=0.2, type=float,
                        help="Percentage of validation data in whole training data. default: 0.2")
    args = parser.parse_args()

    #tf.debugging.set_log_device_placement(True)
    #print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    np.random.seed(args.seed)
    # keras callbacks
    earlystop = EarlyStopping(monitor='val_loss', patience=4) #patience=4
    earlystop_init = EarlyStopping(monitor='val_loss', patience=1)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, cooldown=0, patience=2, min_lr=0.1e-5, verbose=1) #3 #np.sqrt(0.1)
    checkpoint = ModelCheckpoint(args.chkt_filename, monitor='val_loss', mode='min', #f'chkpt_resnet50v2_fold{TESTFOLD}.hdf5'
                                 verbose=0, save_weights_only=False, save_best_only=True)
    checkpoint_init = ModelCheckpoint(args.chkt_filename_init, monitor='val_loss', mode='min', #f'chkpt_resnet50v2_fold{TESTFOLD}_init.hdf5'
                                        verbose=1, save_weights_only=False, save_best_only=True)
                                 
    start_time = time.time()
    tloss_history, taccuracy_history, n_classes, hc_record, labeled_record, \
            pred_labels_np, pred_labels_p, history, tacc_list, vacc_list, tloss_list, vloss_list, \
            epoch_list, end_time_list, perc_x, perc_time, perc_test_acc, cnf_matrices  = run_ceal(args) #cnf_matrix
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    # Report training time and counts (hc and labeled)
    print('Training time {}\n'.format(total_time_str))
    print('wrong_HC_0          : ' + "\t".join([str(x) for x in pred_labels_np]))
    print('wrong_HC_1          : ' + "\t".join([str(x) for x in pred_labels_p]))
    print('HC data count       : ' + "\t".join([str(x) for x in hc_record]))
    print('Labeled data count  : ' + "\t".join([str(x) for x in labeled_record]))
    # Write to textfile
    write_to_textfile_finalise(train_time=total_time_str, hc_np=pred_labels_np, hc_p=pred_labels_p, 
                               hc_total=hc_record, labeled_total=labeled_record, 
                               accuracy_history=taccuracy_history, history=history, 
                               tacc_list=tacc_list, vacc_list=vacc_list, 
                               tloss_list=tloss_list, vloss_list=vloss_list, epoch_list=epoch_list,
                               end_time_list=end_time_list, 
                               perc_x=perc_x, perc_time=perc_time, perc_test_acc=perc_test_acc,
                               cnf_matrices=cnf_matrices,
                               filename=f'finalise{TESTFOLD}_{args.uncertain_criteria}.txt')
    
    # Overall performance
    print('\nAverage accuracy: {avgacc:.3f}'.format(avgacc=np.mean(taccuracy_history)*100))
    plot_graph(performance_history=taccuracy_history, xvals=list(range(0,len(taccuracy_history),1)),
               title='Classification Accuracy', #test accuracy
               ylabel='Accuracy', xlabel='Query iteration', savename='Accuracy_plot', savefig=True)
    #print(perc_x, perc_test_acc, len(perc_x), len(perc_test_acc))
    plot_graph(performance_history=perc_test_acc, xvals=perc_x, title='Test Accuracy', 
               ylabel='Accuracy', xlabel='Percentage of Labeled Data', savename='Perc_Accuracy_plot', savefig=True)
    
