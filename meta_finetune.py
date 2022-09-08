import numpy as np
import pandas as pd
import os
import os.path as osp
import datetime
from GCN import Model, GradientReverseLayer
from util import *
import torch
from sklearn import preprocessing
from scipy.io import loadmat, savemat
import torch.nn.functional as F
import torch.nn as nn
import logging
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import normalize
import random
import copy
import joblib
from util import EEGDataset, GradientReverseLayer, DANet
import torch.utils.data as Data
from sklearn.metrics import roc_auc_score
from braindecode.torch_ext.util import set_random_seeds
from torch.backends import cudnn
import seaborn as sns
from scipy.special import boxcox
from info_nce import InfoNCE

'''
HC label=0  SCZ label=1
All data from huaxi are involved in pre-training, and an equal amount of tuh data is randomly selected 
from each item (all items will cover the full set of tuh)
Two types of tasks are included.
1. domain discrimination
2. binary classification
'''
pre_train = False

if pre_train is True:
    mode = 'wp'
else:
    mode = 'wop'

alpha = 0.3  # discriminator loss weight
beta = 0  # classifier loss weight
# contrastive learning configs
theta = 0.7  # InfoNCE loss weight
n_hard_sample_select = 5
n_positive_sample_select = 5
info_loss_fun = InfoNCE(negative_mode='paired')
num_epochs = 300
#########################################################################
# Load the parameters of the pretrain graph model
if pre_train is True:
    pretrain_file = 'pretrain_50.mat'
    parameters = loadmat(pretrain_file)
    learning_rate = 10 ** float(parameters['learning_rate'])
    l2_regularization = 10 ** float(parameters['l2_regularization'])
    hidden_size = 2 ** int(parameters['hidden_size'])
else:
    learning_rate = 0.001
    l2_regularization = 10 ** float(-3)
    hidden_size = 2 ** int(7)
fea_dim = (16 + 1) * hidden_size // 2

print(f"learning_rate: {learning_rate},"
      f"l2_regularization: {l2_regularization},"
      f"hidden_size: {hidden_size}",
      f"fea_dim: {fea_dim}",
      f"alpha: {alpha}",
      f"beta: {beta}",
      f"theta: {theta}"
      )

# parameters of GCN
time_feature_size = 224
demographic_size = 2
frequency_feature_size = 160

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

# Load the pretrain graph model
if pre_train is True:
    checkpoint_path = 'pretrain_50.pth.tar'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    pretrain_net = Model(adjacency, frequency_feature_size // 16,
                      time_feature_size // 16, demographic_size, hidden_size, 24, 16, device).to(device)
    pretrain_net.load_state_dict(checkpoint)
else:
    pretrain_net = Model(adjacency, frequency_feature_size // 16,
                          time_feature_size // 16, demographic_size, hidden_size, 24, 16, device).to(device)

# load TUH data
psd_features = []
de_features = []
time_features = []
age_and_sex_features = []
path = '../TUH_rawdata/10_2_data'
for i in [x for x in os.listdir(path)]:
    print(i)
    part_folder = os.path.join(os.path.abspath(path), i)
    for j in [y for y in os.listdir(part_folder)]:
        data_path = os.path.join(part_folder, j)
        print(data_path)
        if 'extract_psd_feature' in data_path:
            psd_features.append(np.load(data_path))
        elif 'extract_de_feature' in data_path:
            de_features.append(np.load(data_path))
        elif 'log_extract_time_feature' in data_path:
            time_features.append(np.load(data_path))
        elif 'age_and_sex' in data_path:
            age_and_sex_features.append(np.load(data_path))

psd_features = np.concatenate(psd_features)
de_features = np.concatenate(de_features)
print("DE is nan?: ", np.isnan(tuh_de_features).any())
time_features = np.concatenate(time_features)
age_and_sex_features = np.concatenate(age_and_sex_features)
tuh_psd_features = psd_features
tuh_de_features = de_features
tuh_time_features = time_features
tuh_age_and_sex = age_and_sex_features
tuh_data = np.concatenate((tuh_psd_features.reshape(len(tuh_psd_features), -1),  # psd
                           tuh_de_features.reshape(len(tuh_de_features), -1),  # de
                           tuh_time_features.reshape(len(tuh_time_features), -1),  # time
                           tuh_age_and_sex.reshape(len(tuh_age_and_sex), -1)), axis=-1) # age_and_sex
tuh_label = np.zeros(tuh_data.shape[0])

# load Huaxi data
huaxi_path = "shuffle_feature"
huaxi_psd_features = np.load(huaxi_path + '/patient_psd.npy', allow_pickle=True)
huaxi_de_features = np.load(huaxi_path + '/patient_de.npy', allow_pickle=True)
huaxi_time_features = np.squeeze(np.load(huaxi_path + '/patient_time.npy', allow_pickle=True))
huaxi_age_and_sex = np.load(huaxi_path + '/patient_age_and_sex.npy', allow_pickle=True)

huaxi_data = np.concatenate((huaxi_psd_features.reshape(len(huaxi_psd_features), -1),
                           huaxi_de_features.reshape(len(huaxi_de_features), -1),
                           huaxi_time_features.reshape(len(huaxi_time_features), -1),
                           huaxi_age_and_sex.reshape(len(huaxi_age_and_sex), -1)), axis=-1)
huaxi_label = np.load(huaxi_path + '/patient_label.npy', allow_pickle=True).reshape(-1)[: huaxi_data.shape[0]]
print('参与meta finetune的src_data: ', pd.value_counts(huaxi_label))

model = Model(adjacency, frequency_feature_size // 16,
              time_feature_size // 16, demographic_size, hidden_size, 24, 16, device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_regularization)
dis_criterion = F.nll_loss
clf_criterion = F.nll_loss

best_test_loss = 10 ** 8
for item in range(25):
    """
    seed = 2021
    cuda = True if torch.cuda.is_available() else False
    set_random_seeds(seed=seed, cuda=cuda)
    cudnn.deterministic = True
    """
    src_data = copy.deepcopy(huaxi_data)
    src_label = copy.deepcopy(huaxi_label)
    tuh_data_tmp = copy.deepcopy(tuh_data)
    tuh_label_tmp = copy.deepcopy(tuh_label)

    src_data = src_data.astype('float')

    # Randomly selected data with the same sample size as the huaxi data
    num_sample = src_data.shape[0]
    indices = np.random.choice(np.arange(tuh_data_tmp.shape[0]), replace=False, size=num_sample)
    tar_data = tuh_data_tmp[indices].astype('float')
    tar_label = tuh_label_tmp[indices]
    print("参加meta fine的tuh_data(tst_data)：", tar_data.shape[0])
    
    min_max_scaler = preprocessing.MinMaxScaler()
    src_data = min_max_scaler.fit_transform(src_data)
    tar_data = min_max_scaler.transform(tar_data)
    joblib.dump(min_max_scaler, 'minmax_huaxi_%s' % mode)

    # Mixing tuh data and huaxi data to train the classifier
    all_data = np.concatenate((src_data, tar_data), axis=0)
    all_data = all_data.astype('float')
    all_label = np.concatenate((src_label, tar_label))
    print('all_data：', pd.value_counts(all_label))
    index = [i for i in range(len(all_data))]
    random.shuffle(index)
    all_data = all_data[index]
    all_label = all_label[index]

    # Randomly select num_sample as the training set（80%） and the rest as the test set（20%）
    num_sample = int(0.8 * all_data.shape[0])
    trn_indices = np.random.choice(np.arange(all_data.shape[0]), replace=False, size=num_sample)
    tst_indices = np.delete(range(all_data.shape[0]), trn_indices)
    tst_indices = [i for i in tst_indices]

    trn_data = all_data[trn_indices]
    trn_label = all_label[trn_indices]
    tst_data = all_data[tst_indices]
    tst_label = all_label[tst_indices]
    print('training set:\n', pd.value_counts(trn_label))
    print('testing set:\n', pd.value_counts(tst_label))

    for epoch in range(num_epochs):
        ''' model training'''
        model.train()
        # 1. Domain discrimination task: data without mask
        src_hidden, _, _ = model(
            torch.FloatTensor(src_data[:, :frequency_feature_size].reshape(-1, 16, 10)).to(device),
            torch.FloatTensor(
                src_data[:, frequency_feature_size:frequency_feature_size + time_feature_size].reshape(-1, 16, 14)).to(
                device),
            torch.FloatTensor(src_data[:, frequency_feature_size + time_feature_size:]).to(device))
        discriminate_x_src = model.discriminator(src_hidden, alpha=1)

        tar_hidden, _, _ = model(
            torch.FloatTensor(tar_data[:, :frequency_feature_size].reshape(-1, 16, 10)).to(device),
            torch.FloatTensor(
                tar_data[:, frequency_feature_size:frequency_feature_size + time_feature_size].reshape(-1, 16, 14)).to(
                device),
            torch.FloatTensor(tar_data[:, frequency_feature_size + time_feature_size:]).to(device))
        discriminate_x_tar = model.discriminator(tar_hidden, alpha=1)

        # add domain label
        src_bs = len(src_data)
        source_label = torch.ones(src_bs)
        source_label = source_label.long()

        tar_bs = len(tar_data)
        target_label = torch.zeros(tar_bs)
        target_label = target_label.long()

        domain_output = torch.cat((discriminate_x_src, discriminate_x_tar), dim=0)
        domain_label = torch.cat((source_label, target_label), dim=0)

        if torch.cuda.is_available():
            domain_output = domain_output.cuda()
            domain_label = domain_label.cuda()
        class_weight1 = torch.FloatTensor([1, 1]).to(device)
        trn_dis_loss = dis_criterion(domain_output, domain_label, weight=class_weight1)

        # 2. Classification task: data without mask
        trn_hidden, _, _ = model(
            torch.FloatTensor(trn_data[:, :frequency_feature_size].reshape(-1, 16, 10)).to(device),
            torch.FloatTensor(
                trn_data[:, frequency_feature_size:frequency_feature_size + time_feature_size].reshape(-1, 16,
                                                                                                       14)).to(
                device),
            torch.FloatTensor(trn_data[:, frequency_feature_size + time_feature_size:]).to(device))
        clf_x_train = model.classifier(trn_hidden)
        if torch.cuda.is_available():
            clf_x_train = clf_x_train.cuda()
            trn_label = torch.tensor(trn_label).cuda()
        class_weight2 = torch.FloatTensor([1, 3]).to(device)
        trn_clf_loss = clf_criterion(clf_x_train, trn_label.long(), weight=class_weight2)

        # contrastive loss
        source_feature_health = trn_hidden[trn_label == 0]  # n1, n_fea
        source_feature_patient = trn_hidden[trn_label == 1]  # n2, n_fea
        info_loss = 0
        for i in range(len(source_feature_patient)):
            # select positive key
            positive_keys_list = list(range(0, i)) + list(range(i+1, len(source_feature_patient)))
            random_index = np.random.choice(np.arange(len(positive_keys_list)), replace=False, size=n_positive_sample_select)
            positive_keys_idx = [positive_keys_list[i] for i in random_index]
            query = source_feature_patient[i].unsqueeze(0)  # 1, n_fea
            positive_key = source_feature_patient[positive_keys_idx]  # n_positive_sample_select, n_fea

            # Cosine between query-negative pairs
            dis = torch.sum(query.expand(len(source_feature_health), -1) * source_feature_health, dim=1)
            hard_samples_idx = torch.argsort(dis)[:n_hard_sample_select].reshape(1,-1)
            random_index = np.random.choice(np.arange(n_hard_sample_select, len(source_feature_health)), replace=False, size=n_hard_sample_select)
            random_index = (torch.from_numpy(random_index).reshape(1,-1)).to(device)
            hard_samples_idx = torch.cat((hard_samples_idx, random_index), dim=-1).reshape(-1)

            negative_keys = source_feature_health[hard_samples_idx].unsqueeze(0).expand(len(positive_key), -1, -1)
            info_loss += info_loss_fun(query.expand(len(positive_key), -1), positive_key, negative_keys)
        info_loss /= len(source_feature_patient)

        trn_loss = alpha * trn_dis_loss + beta * trn_clf_loss + theta * info_loss

        optimizer.zero_grad()
        trn_loss.backward()
        optimizer.step()

        '''model testing'''
        model.eval()
        tst_hidden, _, _ = model(
            torch.FloatTensor(tst_data[:, :frequency_feature_size].reshape(-1, 16, 10)).to(device),
            torch.FloatTensor(
                tst_data[:, frequency_feature_size:frequency_feature_size + time_feature_size].reshape(-1, 16, 14)).to(
                device),
            torch.FloatTensor(tst_data[:, frequency_feature_size + time_feature_size:]).to(device))
        clf_x_test = model.classifier(tst_hidden)
        if torch.cuda.is_available():
            clf_x_test = clf_x_test.cuda()
            tst_label = torch.tensor(tst_label).cuda()
        class_weight2 = torch.FloatTensor([1, 2]).to(device)
        test_clf_loss = clf_criterion(clf_x_test, tst_label.long(), weight=class_weight2)

        test_loss = test_clf_loss

        if epoch % 50 == 0:
            print(
                "epoch {:02d} | da_train {:.4f} | clf_train {:.4f} | info_train {:.4f} ｜ loss_train {:.4f} | clf_test {:.4f}"
                .format(epoch, alpha * trn_dis_loss, beta * trn_clf_loss, theta * info_loss, trn_loss,
                        test_clf_loss))

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_test_epoch = epoch
            if epoch % 20 == 0:
                print('Best loss--{:.5f}--at epoch:{}'.format(best_test_loss, best_test_epoch))
            # checkpoint_path = 'finetune1_%s.pth.tar' % mode
            # torch.save(model.state_dict(), checkpoint_path)
print('best_test_loss is: ', best_test_loss)





