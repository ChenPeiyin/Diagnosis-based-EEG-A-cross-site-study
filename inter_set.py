# coding=UTF-8
import numpy as np
import torch
from GCN import Model
from util import EEGDataset, FocalLoss, DANet, GradientReverseLayer
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
import torch.utils.data as Data
from scipy.io import loadmat, savemat
from sklearn import preprocessing
import pandas as pd
import copy
from braindecode.torch_ext.util import set_random_seeds
from torch.backends import cudnn
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score, roc_curve, auc, \
    confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import normalize
from collections import Counter
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization
import logging
import random
import joblib
import seaborn as sns
from scipy.special import boxcox

#  setting
pre_train = True
DA = True
loss_weights = 0.5  # Discriminator loss weights
val_data = 'Hangqi'
ratio = 1  # hangqi data HC:SCZ=ratio
item = 50
num_epoches = 2000
################################################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dis_loss_function = F.nll_loss
if pre_train is True:
    parameters = loadmat('pretrain_50.mat')
    learning_rate = 10 ** float(parameters['learning_rate'])
    l2_regularization = 10 ** float(parameters['l2_regularization'])
    hidden_size = 2 ** int(parameters['hidden_size'])
else:
    learning_rate = 0.01
    l2_regularization = 10 ** float(-3)
    hidden_size = 2 ** int(7)
fea_dim = (16 + 1) * hidden_size // 2

print('discriminator loss weights:', loss_weights)
print('lr:', learning_rate)
print('l2:', l2_regularization)
print("hidden_size", hidden_size)
print("fea_dim", fea_dim)

adjacency = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

adjacency = torch.FloatTensor(np.array(adjacency)).to(device)
time_feature_size = 224
demographic_size = 2
frequency_feature_size = 160

if pre_train is True:
    checkpoint_path = 'finetune1_wp.pth.tar'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    pre_train_net = Model(adjacency, frequency_feature_size // 16,
                          time_feature_size // 16, demographic_size, hidden_size, 24, 16, device).to(device)
    pre_train_net.load_state_dict(checkpoint)
    # hidden_size = pre_train_net.dis[0].in_features
else:
    pre_train_net = Model(adjacency, frequency_feature_size // 16,
                         time_feature_size // 16, demographic_size, hidden_size, 24, 16, device).to(device)
print("hidden_size", hidden_size)
print("fea_dim", fea_dim)

# Huaxi data
path0 = "shuffle_feature"
train_psd_features = np.load(path0 + '/patient_psd.npy', allow_pickle=True)
train_de_features = np.load(path0 + '/patient_de.npy', allow_pickle=True)
train_time_features = np.squeeze(np.load(path0 + '/patient_time.npy', allow_pickle=True))
train_age_and_sex = np.load(path0 + '/patient_age_and_sex.npy', allow_pickle=True)

# Hangqi Data
path1 = "target_shuffle_feature"
test_psd_features1 = np.load(path1 + '/patient_psd.npy', allow_pickle=True)
test_de_features1 = np.load(path1 + '/patient_de.npy', allow_pickle=True)
test_time_features1 = np.squeeze(np.load(path1 + '/patient_time.npy', allow_pickle=True))
test_age_and_sex1 = np.load(path1 + '/patient_age_and_sex.npy', allow_pickle=True)

# Russia Data
path_russia0 = 'russia_hc'
norm_psd_features = np.load(path_russia0 + '/all_psd_feature.npy', allow_pickle=True)
norm_de_features = np.load(path_russia0 + '/all_de_feature.npy', allow_pickle=True)
norm_time_features = np.load(path_russia0 + '/all_time_feature.npy', allow_pickle=True)
norm_age_and_sex = np.load(path_russia0 + '/all_age_and_sex_feature.npy', allow_pickle=True)
norm_label = np.zeros(norm_psd_features.shape[0])

path_russia1 = 'russia_scz'
sch_psd_features = np.load(path_russia1 + '/all_psd_feature.npy', allow_pickle=True)
sch_de_features = np.load(path_russia1 + '/all_de_feature.npy', allow_pickle=True)
sch_time_features = np.load(path_russia1 + '/all_time_feature.npy', allow_pickle=True)
sch_age_and_sex = np.load(path_russia1 + '/all_age_and_sex_feature.npy', allow_pickle=True)
sch_label = np.ones(sch_psd_features.shape[0])

test_psd_features2 = np.concatenate((norm_psd_features, sch_psd_features), axis=0)
test_de_features2 = np.concatenate((norm_de_features, sch_de_features), axis=0)
test_time_features2 = np.concatenate((norm_time_features, sch_time_features), axis=0)
test_age_and_sex2 = np.concatenate((norm_age_and_sex, sch_age_and_sex), axis=0)
test_label2 = np.concatenate((norm_label, sch_label))

# Poland Data
path_wassia0 = 'all_hc_feature'
_norm_psd_features = np.load(path_wassia0 + '/all_psd_feature.npy', allow_pickle=True)
_norm_de_features = np.load(path_wassia0 + '/all_de_feature.npy', allow_pickle=True)
_norm_time_features = np.load(path_wassia0 + '/all_time_feature.npy', allow_pickle=True)
_norm_age_and_sex = np.load('hc_all_age_and_sex_feature.npy', allow_pickle=True)
_norm_label = np.zeros(norm_psd_features.shape[0])

path_wassia1 = 'all_scz_feature'
_sch_psd_features = np.load(path_wassia1 + '/all_psd_feature.npy', allow_pickle=True)
_sch_de_features = np.load(path_wassia1 + '/all_de_feature.npy', allow_pickle=True)
_sch_time_features = np.load(path_wassia1 + '/all_time_feature.npy', allow_pickle=True)
_sch_age_and_sex = np.load('scz_all_age_and_sex_feature.npy', allow_pickle=True)
_sch_label = np.ones(sch_psd_features.shape[0])

test_psd_features3 = np.concatenate((_norm_psd_features, _sch_psd_features), axis=0)
test_de_features3 = np.concatenate((_norm_de_features, _sch_de_features), axis=0)
test_time_features3 = np.concatenate((_norm_time_features, _sch_time_features), axis=0)
test_age_and_sex3 = np.concatenate((_norm_age_and_sex, _sch_age_and_sex), axis=0)
test_label3 = np.concatenate((_norm_label, _sch_label))

train_x = np.concatenate((train_psd_features.reshape(len(train_psd_features), -1),
                          train_de_features.reshape(len(train_de_features), -1),
                          train_time_features.reshape(len(train_time_features), -1),
                          train_age_and_sex.reshape(len(train_age_and_sex), -1)), axis=-1)
train_label = np.load(path0 + '/patient_label.npy', allow_pickle=True).reshape(-1)

test_x1 = np.concatenate((test_psd_features1.reshape(len(test_psd_features1), -1),
                         test_de_features1.reshape(len(test_de_features1), -1),
                         test_time_features1.reshape(len(test_time_features1), -1),
                          test_age_and_sex1.reshape(len(test_age_and_sex1), -1)), axis=-1)
test_label1 = np.load(path1 + '/patient_label.npy', allow_pickle=True).reshape(-1)

test_x2 = np.concatenate((test_psd_features2.reshape(len(test_psd_features2), -1),
                         test_de_features2.reshape(len(test_de_features2), -1),
                         test_time_features2.reshape(len(test_time_features2), -1),
                          test_age_and_sex2.reshape(len(test_age_and_sex2), -1)), axis=-1)
index = [i for i in range(len(test_x2))]
random.shuffle(index)
test_x2 = test_x2[index]
test_label2 = test_label2[index]

test_x3 = np.concatenate((test_psd_features3.reshape(len(test_psd_features3), -1),
                         test_de_features3.reshape(len(test_de_features3), -1),
                         test_time_features3.reshape(len(test_time_features3), -1),
                          test_age_and_sex3.reshape(len(test_age_and_sex3), -1)), axis=-1)
index = [i for i in range(len(test_x3))]
random.shuffle(index)
test_x3 = test_x3[index]
test_label3 = test_label3[index]

# training set
trn_data = train_x
trn_label = train_label
print('source set label: ', pd.value_counts(trn_label))

# testing set
if val_data == "Hangqi":
    # Huaxi -> Hangqi
    # Randomly selected HC and SCZ samples from the Hangqi data according to the proportion
    hc_data = test_x1[test_label1 == 0]
    hc_label = test_label1[test_label1 == 0]
    scz_data = test_x1[test_label1 == 1]
    scz_label = test_label1[test_label1 == 1]
    num_sample = hc_data.shape[0]
    if ratio == 1:  # HC:SCZ = 1：1
        indices = np.random.choice(np.arange(scz_data.shape[0]), replace=False, size=num_sample)
        scz_data = test_x1[test_label1 == 1][indices]
        scz_label = test_label1[test_label1 == 1][indices]
        class_weight2 = torch.FloatTensor([1, 1]).to(device)
    elif ratio == 0.5:  # HC:SCZ = 1：2
        indices = np.random.choice(np.arange(scz_data.shape[0]), replace=False, size=2 * num_sample)
        scz_data = test_x1[test_label1 == 1][indices]
        scz_label = test_label1[test_label1 == 1][indices]
        class_weight2 = torch.FloatTensor([2, 1]).to(device)
    elif ratio == 0.3:  # HC:SCZ = 1：3
        scz_data = scz_data
        scz_label = scz_label
        class_weight2 = torch.FloatTensor([3, 1]).to(device)
    elif ratio == 2:  # HC:SCZ = 2：1
        indices = np.random.choice(np.arange(scz_data.shape[0]), replace=False, size=int(0.5 * num_sample))
        scz_data = test_x1[test_label1 == 1][indices]
        scz_label = test_label1[test_label1 == 1][indices]
        class_weight2 = torch.FloatTensor([1, 2]).to(device)
    elif ratio == 3:  # HC:SCZ = 3：1
        indices = np.random.choice(np.arange(scz_data.shape[0]), replace=False, size=int(0.3 * num_sample))
        scz_data = test_x1[test_label1 == 1][indices]
        scz_label = test_label1[test_label1 == 1][indices]
        class_weight2 = torch.FloatTensor([1, 3]).to(device)
    print("Hangqi SCZ sample：", scz_data.shape[0])

    tst_data = np.concatenate((hc_data, scz_data), axis=0)
    tst_label = np.concatenate((hc_label, scz_label), axis=0)
    index = [i for i in range(len(tst_data))]
    random.shuffle(index)
    tst_data = tst_data[index]
    tst_label = tst_label[index]
    print('Hangqi Data: ', pd.value_counts(tst_label))
    clf_loss_function = F.nll_loss
elif val_data == "Russia":
    # src -> Russia
    tst_data = test_x2
    tst_label = test_label2
    print('Russia Data: ', pd.value_counts(tst_label))
    clf_loss_function = F.nll_loss
    class_weight2 = torch.FloatTensor([1, 1]).to(device)
elif val_data == "Poland":
    # src -> Poland
    tst_data = test_x3
    tst_label = test_label3
    print('Poland Data: ', pd.value_counts(tst_label))
    clf_loss_function = F.nll_loss
    class_weight2 = torch.FloatTensor([1, 1]).to(device)

min_max_scaler = preprocessing.MinMaxScaler()
# min_max_scaler = joblib.load('minmax_huaxi_wp')
trn_data = min_max_scaler.fit_transform(trn_data)
tst_data = min_max_scaler.transform(tst_data)

trn_acc_list = []
trn_f1_list = []
trn_auc_list = []
trn_tpr_list = []
trn_tnr_list = []

tst_auc_list = []
tst_acc_list = []
tst_precision_list = []
tst_f1_list = []
tst_tpr_list = []
tst_tnr_list = []

output_list = []
true_label_list = []

for _item in range(item):

    train_dataset = EEGDataset(trn_data, trn_label)
    test_dataset = EEGDataset(tst_data, tst_label)

    train_loader = Data.DataLoader(train_dataset,
                                   batch_size=32,
                                   shuffle=True)

    test_loader = Data.DataLoader(test_dataset,
                                  batch_size=32,
                                  shuffle=True)
    model = DANet(pre_train_net, frequency_feature_size, time_feature_size, hidden_size, n_class=2).to(device)
  
    optimizer = torch.optim.Adam([
        {'params': model.Fea.parameters(), 'lr': learning_rate, "weight_decay": l2_regularization},
        {'params': model.Classifier.parameters(), 'lr': 0.001, "weight_decay": 1e-3},
        {'params': model.Discrinminator.parameters(), 'lr': 0.001, "weight_decay": 1e-3},
    ])

    best_trn_epoch = 0
    best_trn_acc = 0
    best_trn_auc = 0
    best_trn_precision = 0
    best_trn_f1 = 0
    best_trn_tpr = 0
    best_trn_tnr = 0

    best_tst_epoch = 0
    best_tst_acc = 0
    best_tst_auc = 0
    best_tst_precision = 0
    best_tst_f1 = 0
    best_tst_tpr = 0
    best_tst_tnr = 0

    for epoch in range(num_epoches):
        all_trn_pred = []
        all_trn_true = []
        all_tst_pred = []
        all_tst_true = []
        all_output = []

        """ Training"""
        model.train()

        trn_correct = 0
        test_loader_iter = iter(test_loader)

        for source_data, source_label in train_loader:
            if test_loader_iter:
                try:
                    target_data, target_label = next(test_loader_iter)
                except:
                    test_loader_iter = iter(test_loader)
                    target_data, target_label = next(test_loader_iter)

            source_label = source_label.type(torch.LongTensor)
            target_label = target_label.type(torch.LongTensor)

            source_data = source_data.float().to(device)
            source_label = source_label.to(device)
            target_data = target_data.float().to(device)
            target_label = target_label.to(device)

            optimizer.zero_grad()

            source_feature, target_feature, source_clf, target_clf, source_dis, target_dis = model(source_data,
                                                                                                   target_data)

            clf_loss = clf_loss_function(source_clf, source_label)

            s_pred = source_clf.argmax(axis=1)

            trn_total = len(train_loader.dataset)
            trn_correct += (s_pred == source_label.long()).sum().item()

            all_trn_pred.append(s_pred.detach().cpu().numpy())
            all_trn_true.append(source_label.detach().cpu().numpy())

            source_batch_size = len(source_data)
            source_label = torch.ones(source_batch_size)
            source_label = source_label.long()

            target_batch_size = len(target_data)
            target_label = torch.zeros(target_batch_size)
            target_label = target_label.long()

            domain_output = torch.cat((source_dis, target_dis), axis=0)
            domain_label = torch.cat((source_label, target_label), axis=0)

            domain_output = domain_output.to(device)
            domain_label = domain_label.to(device)

            dis_loss = loss_weights * dis_loss_function(domain_output, domain_label)

            if DA is True:
                loss = clf_loss.cuda() + dis_loss.cuda()
            else:
                loss = clf_loss.cuda()

            loss.backward()
            optimizer.step()

        trn_acc = trn_correct / trn_total
        all_trn_pred = np.concatenate(all_trn_pred).reshape(-1)
        all_trn_true = np.concatenate(all_trn_true).reshape(-1)
        trn_auc = roc_auc_score(all_trn_true, all_trn_pred)
        trn_f1 = f1_score(all_trn_pred, all_trn_true)
        cm = confusion_matrix(all_trn_true, all_trn_pred)
        tn, fp, fn, tp = confusion_matrix(all_trn_true, all_trn_pred).ravel()
        trn_tpr = tp / (tp + fn)
        trn_tnr = tn / (tn + fp)

        trn_thresholds = np.abs(trn_tpr - trn_tnr)

        if trn_auc >= best_trn_auc:
            if trn_thresholds <= 0.3:
                best_trn_epoch = epoch
                best_trn_acc = trn_acc
                best_trn_auc = trn_auc
                best_trn_f1 = trn_f1
                best_trn_tpr = trn_tpr
                best_trn_tnr = trn_tnr
            else:
                pass

        if epoch % 50 == 0:
            print(
                "Train current epoch: {}, trn_clf_loss: {:.5f}, trn_dis_loss: {:.5f}, trn_total_loss: {:.5f}, trn_clf_acc: {}/{} ({:.5f})\n".
                format(epoch, clf_loss, dis_loss, loss, trn_correct, trn_total, trn_acc))
            print(
                "Best train accuracy / F1-score / AUC / TPR / TNR at epoch {} : {:.5f} / {:.5f} / {:.5f} / {:.5f} / {:.5f}".format(
                    best_trn_epoch, best_trn_acc, best_trn_f1, best_trn_auc, best_trn_tpr, best_trn_tnr))

        """ Testing"""
        model.eval()
        tst_correct = 0

        for i, (target_data, target_label) in enumerate(test_loader):
            target_data = target_data.float().to(device)
            target_label = target_label.to(device)

            tst_feature0, tst_feature, tst_clf0, tst_clf, tst_dis0, tst_dis = model(target_data, target_data)

            tst_pred = tst_clf.argmax(axis=1)
            tst_total = len(test_loader.dataset)
            tst_correct += (tst_pred == target_label.long()).sum().item()

            class_weight2 = class_weight2
            tst_clf_loss = clf_loss_function(tst_clf, target_label.long(), weight=class_weight2)

            all_output.append(tst_clf0.detach().cpu().numpy())
            all_tst_pred.append(tst_pred.detach().cpu().numpy())
            all_tst_true.append(target_label.detach().cpu().numpy())

        tst_acc = tst_correct / tst_total
        all_tst_pred = np.concatenate(all_tst_pred).reshape(-1)
        all_tst_true = np.concatenate(all_tst_true).reshape(-1)
        tst_auc = roc_auc_score(all_tst_true, all_tst_pred)
        tst_f1 = f1_score(all_tst_pred, all_tst_true)
        tst_precision = precision_score(all_tst_true, all_tst_pred)
        cm = confusion_matrix(all_tst_true, all_tst_pred)
        tn, fp, fn, tp = confusion_matrix(all_tst_true, all_tst_pred).ravel()
        tst_tpr = tp / (tp + fn)
        tst_tnr = tn / (tn + fp)
        tst_thresholds = np.abs(tst_tpr - tst_tnr)

        if epoch % 50 == 0:
            print('\nTest set: tst_clf_loss: {:.5f}, tst_clf_accuracy: {}/{} ({:.5f}), tst_f1: {:5f}\n'.format(
                tst_clf_loss, tst_correct,
                tst_total, tst_acc, tst_f1))
            print(pd.value_counts(all_tst_pred))
            print(pd.value_counts(all_tst_true))

        if tst_auc >= best_tst_auc:
            if tst_thresholds <= 0.3:
                best_tst_epoch = epoch
                best_tst_acc = tst_acc
                best_tst_auc = tst_auc
                best_tst_precision = tst_precision
                best_tst_f1 = tst_f1
                best_tst_tpr = tst_tpr
                best_tst_tnr = tst_tnr
                output = all_output
                true_label = all_tst_true
            else:
                pass
            checkpoint_path = 'msu_model4.pth.tar'
            torch.save(model.state_dict(), checkpoint_path)

        if epoch % 50 == 0:
            print('Best auc--{:.5f}--acc--{:.5f}--precision--{:.5f}--f1--{:.5f}\n'
                  'tpr--{:.5f}--tnr--{:.5f} at epoch:{}'.format(best_tst_auc,
                                                               best_tst_acc,
                                                               best_tst_precision,
                                                               best_tst_f1,
                                                               best_tst_tpr,
                                                               best_tst_tnr, best_tst_epoch))
            
            print("****************************************************************************")

    trn_auc_list.append(best_trn_auc)
    trn_acc_list.append(best_trn_acc)
    trn_f1_list.append(best_trn_f1)
    trn_tpr_list.append(best_trn_tpr)
    trn_tnr_list.append(best_trn_tnr)

    tst_auc_list.append(best_tst_auc)
    tst_acc_list.append(best_tst_acc)
    tst_precision_list.append(best_tst_precision)
    tst_tpr_list.append(best_tst_tpr)
    tst_f1_list.append(best_tst_f1)
    tst_tnr_list.append(best_tst_tnr)

    output_list.append(output)
    true_label_list.append(true_label)
    print("current item is finished: {}".format(_item + 1))

mean_trn_auc = np.mean(trn_auc_list)
MSE_trn_auc = np.std(trn_auc_list)

mean_trn_acc = np.mean(trn_acc_list)
MSE_trn_acc = np.std(trn_acc_list)

mean_trn_f1 = np.mean(trn_f1_list)
MSE_trn_f1 = np.std(trn_f1_list)

mean_trn_tpr = np.mean(trn_tpr_list)
MSE_trn_tpr = np.std(trn_tpr_list)

mean_trn_tnr = np.mean(trn_tnr_list)
MSE_trn_tnr = np.std(trn_tnr_list)

print("Mean train AUC: {:.5f}±{:.5f}\n"
      "Mean train ACC: {:.5f}±{:.5f}\n"
      "Mean train F1: {:.5f}±{:.5f}\n"
      "Mean train TPR: {:.5f}±{:.5f}\n"
      "Mean train TNR: {:.5f}±{:.5f}".format(mean_trn_auc, MSE_trn_auc,
                                              mean_trn_acc, MSE_trn_acc,
                                              mean_trn_f1, MSE_trn_f1,
                                              mean_trn_tpr, MSE_trn_tpr,
                                              mean_trn_tnr, MSE_trn_tnr))

mean_tst_auc = np.mean(np.array(tst_auc_list))
MSE_auc = np.std(tst_auc_list)

mean_tst_acc = np.mean(np.array(tst_acc_list))
MSE_acc = np.std(tst_acc_list)

mean_tst_precision = np.mean(np.array(tst_precision_list))
MSE_precision = np.std(tst_precision_list)

mean_tst_f1 = np.mean(np.array(tst_f1_list))
MSE_f1 = np.std(tst_f1_list)

mean_tst_tpr = np.mean(np.array(tst_tpr_list))
MSE_tpr = np.std(tst_tpr_list)

mean_tst_tnr = np.mean(np.array(tst_tnr_list))
MSE_tnr = np.std(tst_tnr_list)

print("Current inter_set experiment is :lr = %s, loss_weights = %s, pre_train = %s, DA = %s, val_data = %s" % (learning_rate, loss_weights, pre_train, DA, val_data))
print("seed_list: ", seed_list)
print("ratio: ", ratio)
print("tst_auc_list: ", tst_auc_list)
print("tst_acc_list: ", tst_acc_list)
print("tst_f1_list: ", tst_f1_list)
print("tst_tpr_list: ", tst_tpr_list)
print("tst_tnr_list: ", tst_tnr_list)
print("Mean test AUC: {:.3f}±{:.3f}\n"
      "Mean test ACC: {:.3f}±{:.3f}\n"
      "Mean test precision: {:.3f}±{:.3f}\n"
      "Mean test F1: {:.3f}±{:.3f}\n"
      "Mean test TPR: {:.3f}±{:.3f}\n"
      "Mean test TNR: {:.3f}±{:.3f}".format(mean_tst_auc, MSE_auc,
                                          mean_tst_acc, MSE_acc,
                                          mean_tst_precision, MSE_precision,
                                          mean_tst_f1, MSE_f1,
                                          mean_tst_tpr, MSE_tpr,
                                          mean_tst_tnr, MSE_tnr))








