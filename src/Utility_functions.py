import natsort
import random
import glob
import os
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras import layers
from sklearn.utils import class_weight
import numpy as np
random.seed(32)
import pandas as pd
from tensorflow.python.keras.callbacks import Callback
def downsample_withoutshuffle(f_list, fpvpb, no_frames):
    f_list = natsort.natsorted(f_list)
    _total = int(no_frames / fpvpb)  # 20/5
    _tempList = []
    # print("Size:",_total)
    if len(f_list) > _total:  # if list smaller than required frames
        for i in range(0, _total):
            _range = len(f_list) - fpvpb
            if _range == 0:
                return f_list
            x = random.randrange(_range)
            for j in range(0, fpvpb):
                # print(os.path.splitext(os.path.basename(f_list[x+j]))[0],end=",")
                _tempList.append(f_list[x + j])
            # print()
        return _tempList
    else:
        return f_list

def create_sequence(dirs, fpvpb, no_frames):
    random.seed(35)
    count_real = 0
    count_fake = 0
    folders=[]
    for directory in dirs:
        folders += [i for i in sorted(glob.glob(os.path.join(directory, '*', "*")))]
    random.shuffle(folders)
    total_folders = len(folders)
    X = {}
    y = []
    print('Total Video Folders Found (Real + Fake):', total_folders)
    pre_folder = -1
    i = 0
    _downsampled = []
    while i < total_folders:
        folder = random.choice(folders)
        dir_name = os.path.dirname(folder)
        folder_name = os.path.basename(dir_name)
        #         print(folder_name)
        _downsampled = [x for x in sorted(glob.glob(os.path.join(folder, "*")))][0:no_frames]
        #         _downsampled=downsample_withoutshuffle([x for x in sorted(glob.glob(os.path.join(folder,"*")))],fpvpb,no_frames)
        if folder_name == 'real':
            if pre_folder == 1:
                continue
            y.append(1)
            count_real += len(_downsampled)
            pre_folder = 1
        elif folder_name == 'fake':
            if pre_folder == 0:
                continue
            y.append(0)
            count_fake += len(_downsampled)
            pre_folder = 0
        else:
            print("Directory names should be 'real' for label (1) and 'fake' for label (0)")
            exit(0)
        # X[str(i)]=[x for x in sorted(glob.glob(os.path.join(folder,"*")))]
        X[str(i)] = _downsampled
        folders.remove(folder)
        i += 1
    print('Real Frames:', count_real, 'Fake Frames:', count_fake)
    labels = []
    for i in range(0, count_real):
        labels.append(1)
    for i in range(0, count_fake):
        labels.append(0)
    y_ints = [v.argmax() for v in to_categorical(labels)]
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_ints), y_ints)
    print('Class Weights:', class_weights)
    return X, y, class_weights

class FrozenBatchNormalization(layers.BatchNormalization):
    def call(self, inputs, training=None):
        return super().call(inputs=inputs, training=False)
def FreezeBatchNormalization(is_training,top_k_layers,model):
    if is_training == True:
        _bottom_layers = model.layers[:-top_k_layers]
        _top_layers = model.layers[-top_k_layers:]
    elif is_training == False:
        _bottom_layers = model.layers
        _top_layers = []

    for _layer in _bottom_layers:
        _layer.trainable = False
        if 'batch_normalization' in _layer.name:
    #         print('Freezing BN layers ... {}'.format(_layer.name))
            _layer = FrozenBatchNormalization

    for _layer in _top_layers:
        _layer.trainable = True
        if 'batch_normalization' in _layer.name:
    #         print('Unfreezing BN layers ... {}'.format(_layer.name))
            _layer = layers.BatchNormalization

    layers_df = [(layer, layer.name, layer.trainable) for layer in model.layers]
    df=pd.DataFrame(layers_df, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
    return model,df



class AdditionalValidationSets(Callback):
    def __init__(self, validation_gen_sets,filename, verbose=1, batch_size=None):
        """
        :param validation_sets:
        a list of 3-tuples (validation_data, validation_targets, validation_set_name)
        or 4-tuples (validation_data, validation_targets, sample_weights, validation_set_name)
        :param verbose:
        verbosity mode, 1 or 0
        :param batch_size:
        batch size to be used when evaluating on the additional datasets
        """
        super(AdditionalValidationSets, self).__init__()
        self.validation_gen_sets = validation_gen_sets
#         for validation_set in self.validation_sets:
#             if len(validation_set) not in [2, 3]:
#                 raise ValueError()
        self.epoch = []
        self.history = {}
        self.verbose = verbose
        self.batch_size = batch_size
        self.filename = filename
        self.previous_data=""
        self.save_previous=False

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}
        if os.path.exists(self.filename):
            self.save_previous=True
            with open(self.filename, 'r') as file:
                self.previous_data = file.read()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
#         line=str(epoch)+","
        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # evaluate on the additional validation sets
        for validation_gen_set in self.validation_gen_sets:
            if len(validation_gen_set) == 2:
                validation_gen, validation_set_name = validation_gen_set
#                     sample_weights = None
#             elif len(validation_set) == 4:
#                 validation_data, validation_targets, sample_weights, validation_set_name = validation_set
#             else:
#                 raise ValueError()
            
            results = self.model.evaluate_generator(validation_gen,
                                          verbose=self.verbose)
#                                           sample_weight=sample_weights,
#                                           batch_size=self.batch_size)
            
            for i, result in enumerate(results):
                if i == 0:
                    valuename = validation_set_name + '_loss'
                else:
                    valuename = str(validation_set_name) + '_acc'#acc + str(self.model.metrics[i-1])
                self.history.setdefault(valuename, []).append(result)
        line=""
        previous=0
        if self.save_previous==False:
            line="epoch,"
            dic=self.history
            for item in dic:
                line+=item+","
            line=line[:-1]+"\n"
        else:
            line=self.previous_data
            previous=len(line.splitlines())
        for i in range(0,len(dic['loss'])):
            line+=str(i+1+previous)+","
            for item in dic:
                line+=str(dic[item][i])+","
            line=line[:-1]+"\n"
        with open(self.filename, 'w') as the_file:
            the_file.write(line)