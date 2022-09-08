import numpy as np
import pandas as pd
import os
import os.path as osp
import datetime
from GCN import Model, GradientReverseLayer
from util import *
import torch
from sklearn import preprocessing
from scipy.io import savemat
import torch.nn.functional as F
import torch.nn as nn
import logging
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import normalize
import random
import joblib

'''
tuh_set is used for pretraining, repeating the experiment 50 times
'''
def pre_train_GCN(learning_rate, l2_regularization, hidden_size):
    learning_rate = 10 ** learning_rate
    l2_regularization = 10 ** l2_regularization
    hidden_size = 2 ** int(hidden_size)
    num_epochs = 300

    print(f"learning_rate: {learning_rate},"
          f"l2_regularization: {l2_regularization},"
          f"hidden_size: {hidden_size}"
          )

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
    masked_index = np.random.randint(0, 16)  # masked channel

    psd_features = []
    de_features = []
    time_features = []
    age_and_sex_features = []
    path = r'C:\Users\niyiepnehC\Desktop\交接\TUH_data'
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
            elif 'all_age_and_sex' in data_path:
                age_and_sex_features.append(np.load(data_path))

    psd_features = np.concatenate(psd_features)
    de_features = np.concatenate(de_features)
    time_features = np.concatenate(time_features)
    age_and_sex_features = np.concatenate(age_and_sex_features)

    np.save(r'C:\Users\niyiepnehC\Desktop\交接\tuh1w\extract_psd_feature.npy', psd_features)
    np.save(r'C:\Users\niyiepnehC\Desktop\交接\tuh1w\extract_de_feature.npy', de_features)
    np.save(r'C:\Users\niyiepnehC\Desktop\交接\tuh1w\extract_time_feature.npy', time_features)
    np.save(r'C:\Users\niyiepnehC\Desktop\交接\tuh1w\all_age_and_sex.npy', age_and_sex_features)

    # Randomly select num_sample as the training set（80%） and the rest as the test set（20%）
    num_sample = 2500
    trn_indices = np.random.choice(np.arange(psd_features.shape[0]), replace=False, size=num_sample)
    tst_indices = np.delete(range(psd_features.shape[0]), trn_indices)
    tst_indices = [i for i in tst_indices]
    train_psd_features = psd_features[trn_indices]
    train_de_features = de_features[trn_indices]
    train_time_features = time_features[trn_indices]
    train_age_and_sex = age_and_sex_features[trn_indices]

    test_psd_features = psd_features[tst_indices]
    test_de_features = de_features[tst_indices]
    test_time_features = time_features[tst_indices]
    test_age_and_sex = age_and_sex_features[tst_indices]

    print(np.isnan(train_de_features).any())
    print(np.isnan(test_de_features).any())
    
    train_label = np.concatenate((train_psd_features[:, masked_index, :],
                                   train_de_features[:, masked_index, :],
                                   train_time_features[:, masked_index, :]), axis=-1)
    test_label = np.concatenate((test_psd_features[:, masked_index, :],
                                  test_de_features[:, masked_index, :],
                                  test_time_features[:, masked_index, :]), axis=-1)

    train_psd_features[:, masked_index, :] *= 0
    train_de_features[:, masked_index, :] *= 0 
    train_time_features[:, masked_index, :] *= 0

    test_psd_features[:, masked_index, :] *= 0
    test_de_features[:, masked_index, :] *= 0
    test_time_features[:, masked_index, :] *= 0

    train_x = np.concatenate((train_psd_features.reshape(len(train_psd_features), -1),
                              train_de_features.reshape(len(train_de_features), -1),
                              train_time_features.reshape(len(train_time_features), -1),
                              train_age_and_sex.reshape(len(train_age_and_sex), -1)), axis=-1)

    test_x = np.concatenate((test_psd_features.reshape(len(test_psd_features), -1),
                             test_de_features.reshape(len(test_de_features), -1),
                             test_time_features.reshape(len(test_time_features), -1),
                             test_age_and_sex.reshape(len(test_age_and_sex), -1)), axis=-1)
   
    model = Model(adjacency, frequency_feature_size // 16,
                  time_feature_size // 16, demographic_size, hidden_size, 24, 16, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_regularization)
    loss_function = nn.L1Loss(reduction='mean')
   
    best_loss = 10**9
    for epoch in range(num_epochs):
        ''' model training'''
        model.train()
        
        min_max_scaler = preprocessing.MinMaxScaler()
        train_x = min_max_scaler.fit_transform(train_x)
        test_x = min_max_scaler.transform(test_x)
        joblib.dump(min_max_scaler, 'min_max_scalar_tuh')

        # 1. Prediction task: data with mask
        train_hidden, _, _ = model(
            torch.FloatTensor(train_x[:, :frequency_feature_size].reshape(-1, 16, 10)).to(device),
            torch.FloatTensor(
                train_x[:, frequency_feature_size:frequency_feature_size + time_feature_size].reshape(-1, 16, 14)).to(
                device),
            torch.FloatTensor(train_x[:, frequency_feature_size + time_feature_size:]).to(device))
        # print(train_hidden_m.size())
        generated_x_train = model.generate(train_hidden)
        # print('generated_x_train shape: ', generated_x_train.shape)
        train_loss_mse = F.mse_loss(generated_x_train, torch.FloatTensor(train_label).to(device), reduction='mean')
        train_loss_mae = loss_function(generated_x_train, torch.FloatTensor(train_label).to(device))
        train_loss = train_loss_mse + train_loss_mae

        optimizer.zero_grad()
        train_loss.backward(retain_graph=True)
        optimizer.step()

        '''model testing'''
        model.eval()
        test_hidden, _, _ = model(torch.FloatTensor(test_x[:, :frequency_feature_size].reshape(-1, 16, 10)).to(device),
                                  torch.FloatTensor(test_x[:,
                                                    frequency_feature_size:frequency_feature_size + time_feature_size].reshape(-1, 16,
                                                                                                                 14)).to(device),
                                  torch.FloatTensor(test_x[:, frequency_feature_size + time_feature_size:]).to(device))
        generated_x_test = model.generate(test_hidden)
        test_pre_loss_mae = loss_function(generated_x_test, torch.FloatTensor(test_label).to(device))
        test_pre_loss_mse = F.mse_loss(generated_x_test, torch.FloatTensor(test_label).to(device), reduction='mean')

        if epoch % 20 == 0:
            print("epoch {:02d} | mse_train {:.4f} | mae_train {:.4f} | mse_test {:.4f} | mae_test {:.4f} |"
                     .format(epoch, train_loss_mse, train_loss_mae, test_pre_loss_mae, test_pre_loss_mse))

        # test_loss = test_pre_loss_mae + test_pre_loss_mse
        test_loss = test_pre_loss_mse
        if test_loss < best_loss:
            best_loss = test_loss
            checkpoint_path = 'pretrain_50.pth.tar'
            torch.save(model.state_dict(), checkpoint_path)
        # best_loss = torch.from_numpy(np.array(best_loss)).cpu()
    return -best_loss.detach().cpu().numpy()

if __name__ == '__main__':
    save_logging('pretrain_50.log')
    log_file = '%s.txt' % datetime.date.today()
    working_dir = osp.dirname(osp.abspath(__file__))

    logs_dir = osp.join(working_dir, 'logs')
    if not osp.isdir(logs_dir):
        os.makedirs(logs_dir)

    reward_train = BayesianOptimization(
        pre_train_GCN, {
            'hidden_size': (4, 7),
            'learning_rate': (-5, -2),
            'l2_regularization': (-5, -1)
        }
    )
    reward_train.maximize(n_iter=50)
    logging.info(reward_train.max)

    out = pre_train_GCN(reward_train.max['params']['learning_rate'],
                        reward_train.max['params']['l2_regularization'],
                        reward_train.max['params']['hidden_size'])

    file_name = 'pretrain_50.mat'
    savemat(file_name, {'learning_rate': reward_train.max['params']['learning_rate'],
                        'l2_regularization': reward_train.max['params']['l2_regularization'],
                        'hidden_size': reward_train.max['params']['hidden_size']
    })




