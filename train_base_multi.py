#!/usr/bin/env python
# coding: utf-8
'''Subject-independent classification with KU Data,
using Deep ConvNet model from [1].

References
----------
.. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
   Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
   Deep learning with convolutional neural networks for EEG decoding and
   visualization.
   Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
'''

import argparse
import json
import logging
import pickle
import sys
from os import makedirs
from os.path import join as pjoin
from shutil import copy2, move

import h5py
import numpy as np
import sklearn.metrics
import torch
import torch.nn.functional as F
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.models.deep4 import Deep4Net
from braindecode.models.eegnet import EEGNetv4
from braindecode.torch_ext.optimizers import AdamW
from braindecode.torch_ext.util import set_random_seeds
from sklearn.model_selection import KFold


def get_args():
    parser = argparse.ArgumentParser(
        description='Subject-independent classification with MultiClass KU Data')
    parser.add_argument('datapath', type=str, help='Path to the h5 data file')
    parser.add_argument('outpath', type=str, help='Path to the result folder')
    parser.add_argument('-fold', type=int, default=0,
                        help='5-fold index, starts with 0')
    parser.add_argument('-gpu', type=int, help='The gpu device to use', default=0)
    parser.add_argument("--model", choices=[0, 1], type=int, help="0: DeepConvNet, 1:EEGNetv4")
    parser.add_argument("--move_type", choices=[0, 1], type=int, help="0: MI, 1: realMove")
    return parser.parse_args()

def convert_labels_to_numerals(ydata):
    ydata_numeral = np.zeros((len(ydata),))
    for i, label in enumerate(labels_for_classif):
        ydata_numeral[np.where(ydata==label)] = i
    return np.int64(ydata_numeral)

def get_data_multiclass(subj, move_type):
    eeg_fname = f"Sub{subj}_{move_type}.npy"
    eegdata = np.load(pjoin(datapath,eeg_fname),allow_pickle=True).item()
    xdata, ylabels = eegdata['xdata'], eegdata['yLabels']
    ylabels_stripped = np.array([label.translate(str.maketrans("","","[']")) for label in ylabels])
    idx_chosen = [i for i in range(len(ylabels_stripped)) if ylabels_stripped[i] in labels_for_classif]
    ydata = convert_labels_to_numerals(ylabels_stripped[idx_chosen])
    xdata = xdata[idx_chosen]
    return xdata, ydata

# def get_data(subj):
    # dpath = '/s' + str(subj)
    # X = dfile[pjoin(dpath, 'X')]
    # Y = dfile[pjoin(dpath, 'Y')]
    # return X, Y


def get_multi_data(subjs, move_type):
    Xs = []
    Ys = []
    for s in subjs:
        x, y = get_data_multiclass(s, move_type)
        Xs.append(x[:])
        Ys.append(y[:])
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    return X, Y


def experiment(subjs, test_subj, labels_idx_for_classif, outpath, model_choice, move_type):
    cv_loss = []
    torch.cuda.set_device(args.gpu)
    set_random_seeds(seed=20200205, cuda=True)
    BATCH_SIZE = 16
    TRAIN_EPOCH = 200  # consider 200 for early stopping
    train_subjs = subjs[subjs != test_subj]
    # valid_subjs = cv_set[test_index]
    X_train, Y_train = get_multi_data(train_subjs, move_type)
    # X_val, Y_val = get_multi_data(valid_subjs)
    X_test, Y_test = get_data_multiclass(test_subj, move_type)
    train_set = SignalAndTarget(X_train, y=Y_train)
    # valid_set = SignalAndTarget(X_val, y=Y_val)
    test_set = SignalAndTarget(X_test, y=Y_test)
    n_classes = len(labels_idx_for_classif)
    in_chans = train_set.X.shape[1]

    # final_conv_length = auto ensures we only get a single output in the time dimension
    if model_choice == 0:
        model = Deep4Net(in_chans=in_chans, n_classes=n_classes,
                     input_time_length=train_set.X.shape[2],
                     final_conv_length='auto').cuda()
    else:
        model = EEGNetv4(in_chans=in_chans, n_classes=n_classes, input_time_length=train_set.X.shape[2],
                         final_conv_length="auto", pool_mode="mean", kernel_length=125).cuda()
    # these are good values for the deep model
    optimizer = AdamW(model.parameters(), lr=1*0.01, weight_decay=0.5*0.001)
    model.compile(loss=F.nll_loss, optimizer=optimizer, iterator_seed=1, )

    # Fit the base model for transfer learning, use early stopping as a hack to remember the model
    # exp = model.fit(train_set.X, train_set.y, epochs=TRAIN_EPOCH, batch_size=BATCH_SIZE, scheduler='cosine',
    #                 validation_data=(valid_set.X, valid_set.y), remember_best_column='valid_loss')
    exp = model.fit(train_set.X, train_set.y, epochs=TRAIN_EPOCH, batch_size=BATCH_SIZE, scheduler='cosine',
                    validation_data=None, remember_best_column=None)

    rememberer = exp.rememberer
    base_model_param = {
        # 'epoch': rememberer.best_epoch,
        # 'model_state_dict': rememberer.model_state_dict,
        'model_state_dict': exp.model.state_dict(),
        # 'optimizer_state_dict': rememberer.optimizer_state_dict,
        # 'loss': rememberer.lowest_val
    }

    # torch.save(base_model_param, pjoin(
    #     outpath, 'model_f{}_cv{}.pt'.format(fold, cv_index)))
    torch.save(base_model_param, pjoin(
        outpath, f"model_{test_subj}.pt"))
    # model.epochs_df.to_csv(
    #     pjoin(outpath, 'epochs_f{}_cv{}.csv'.format(fold, cv_index)))
    model.epochs_df.to_csv(
        pjoin(outpath, f"epochs_{test_subj}.csv"))

    # cv_loss.append(rememberer.lowest_val)

    test_loss = model.evaluate(test_set.X, test_set.y)
    print(test_loss)
    test_labels_pred = model.predict_classes(test_set.X)
    conf_matr = sklearn.metrics.confusion_matrix(test_set.y,test_labels_pred)
    test_accuracy = (np.sum(test_labels_pred==test_set.y))/len(test_labels_pred)
    test_loss['conf_matr'] = conf_matr
    test_loss['test_accr'] = test_accuracy
    # with open(pjoin(outpath, 'test_base_s{}_f{}_cv{}.json'.format(test_subj, fold, cv_index)), 'w') as f:
    #     json.dump(test_loss, f)
    with open(pjoin(outpath, f"test_base_{test_subj}.pickle"), 'wb') as f:
        pickle.dump(test_loss, f)



if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                        level=logging.INFO, stream=sys.stdout)
    args = get_args()
    datapath = args.datapath
    outpath = args.outpath
    fold = args.fold
    assert (fold >= 0 and fold < 5)
    MODELS = ("DCNet", "EEGNet")
    print(f"Model is {MODELS[args.model]}")
    subjs = np.array(range(1, 26))
    ALL_MOVE = ("MI", "realMove")
    move_type = ALL_MOVE[args.move_type]
    print(f"Move type {move_type}")
    UNIQUE_LABELS = ["all", "Backward", "Cylindrical", "Down", "Forward", "Left", "Lumbrical", "Right", "Spherical",
                     "Up", "twist_Left", "twist_Right"]
    # labels_idx_for_classif = [2, 4, 10, 11]
    # labels_idx_for_classif = list(range(1,12))
    labels_idx_for_classif = [1, 4]
    labels_for_classif = [UNIQUE_LABELS[idx] for idx in labels_idx_for_classif]
    print(f"Labels chosen for multi-class classification = {labels_for_classif}")

    cv_set = range(fold * 5 + 1, (fold + 1) * 5 + 1)
    print(f"Test subjects in this CV: {[cv_set]}")

    for test_subj in cv_set:
        print(f"Test Subject {test_subj}")
        experiment(subjs, test_subj, labels_idx_for_classif, outpath, args.model, move_type)