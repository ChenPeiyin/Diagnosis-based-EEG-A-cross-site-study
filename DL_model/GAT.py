import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from save_results import save_result
from bayes_opt import BayesianOptimization
from util import *
from torch_geometric.nn import GATConv
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from evaluate_model import evaluate_model_performance_two_class
import warnings
warnings.filterwarnings("ignore")


class Model(nn.Module):
    def __init__(self, frequency_feature_size,
                 time_feature_size, demographic_size, hidden_1, output_size, channel, device):
        super(Model, self).__init__()
        self.device = device

        self.gcn1 = GATConv(frequency_feature_size, hidden_1).to(device)
        self.gcn2 = GATConv(hidden_1, hidden_1).to(device)

        self.gcn3 = GATConv(time_feature_size, hidden_1).to(device)
        self.gcn4 = GATConv(hidden_1, hidden_1).to(device)

        self.fc1 = nn.Linear(demographic_size, hidden_1)
        self.output_1 = nn.Linear((2 * channel + 1) * hidden_1, output_size)

    def forward(self, input_frequency, input_time_domain, demographics, edge_index):
        hidden_frequency = torch.relu(self.gcn1(input_frequency, edge_index))
        hidden_frequency = F.dropout(hidden_frequency, p=0.6, training=self.training)
        hidden_frequency = self.gcn2(hidden_frequency, edge_index)
        hidden_frequency = F.dropout(hidden_frequency, p=0.6, training=self.training)

        hidden_time = torch.relu(self.gcn3(input_time_domain, edge_index))
        hidden_time = F.dropout(hidden_time, p=0.6, training=self.training)
        hidden_time = self.gcn4(hidden_time, edge_index)
        hidden_time = F.dropout(hidden_time, p=0.6, training=self.training)

        hidden_demographic = torch.relu(self.fc1(demographics))
        hidden_demographic = torch.unsqueeze(hidden_demographic, dim=0)
        hidden_all = torch.cat((hidden_frequency, hidden_time, hidden_demographic), dim=0)
        hidden_all = hidden_all.reshape(1, -1)
        output = self.output_1(hidden_all)
        return output, torch.sigmoid(output)


def run_experiment(index, learning_rate, hidden_1):
    path = '..'
    labels = np.zeros(shape=[0, 1])
    predicted_labels_prob = np.zeros(shape=[0, 1])

    features_ = np.load(path + '/all_patient_frequency_features_15360.npy').reshape(-1, 16, 5)
    features_de = np.load(path + '/all_patient_de_features_15360.npy').reshape(-1, 16, 5)
    features_ = np.concatenate((features_, features_de,
                                ), axis=1)
    if index == -1:
        features_ = features_[:, :, :]
        frequency_feature_size = 160
        channel = 16
    else:
        features_ = features_[:, :, index]
        frequency_feature_size = 32
        channel = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    time_feature_size = 224
    demographic_size = 2
    edge_index = np.array([[0, 1], [1, 0], [2, 3], [3, 2], [4, 5], [5, 4], [6, 7], [7, 6],
                           [8, 9], [9, 8]]).transpose(1, 0)

    labels_ = np.load(path + '/all_patient_frequency_labels_15360.npy').reshape(-1, )
    age_and_sex_features = np.load(path + '/all_patient_age_and_sex.npy').reshape(-1, 2)
    time_mean_and_std = np.load(path + '/all_patient_time_features.npy').reshape(-1, 16, 14)

    hc_index = 1
    scz_index = 2
    labels_hc_index = np.argwhere(labels_ == hc_index)
    features_hc = features_[labels_hc_index]
    age_and_sex_features_hc = age_and_sex_features[labels_hc_index]
    time_mean_and_std_hc = time_mean_and_std[labels_hc_index]
    features_hc = np.concatenate(
        (features_hc.reshape(len(features_hc), -1),
         time_mean_and_std_hc.reshape(len(time_mean_and_std_hc), -1),
         age_and_sex_features_hc.reshape(len(age_and_sex_features_hc), -1),
         ),
        axis=1)
    labels_hc = np.zeros(shape=[len(labels_hc_index)])

    labels_scz_index = np.argwhere(labels_ == scz_index)
    features_scz = features_[labels_scz_index]
    age_and_sex_features_scz = age_and_sex_features[labels_scz_index]
    time_mean_and_std_scz = time_mean_and_std[labels_scz_index]
    features_scz = np.concatenate((features_scz.reshape(len(features_scz), -1),
                                   time_mean_and_std_scz.reshape(len(time_mean_and_std_scz), -1),
                                   age_and_sex_features_scz.reshape(len(age_and_sex_features_scz), -1),
                                   ), axis=1)
    labels_scz = np.ones(shape=[len(labels_scz_index)])

    # 平衡数据集
    # patient_min_length = min(len(features_hc), len(features_scz))
    # features_hc = features_hc[:patient_min_length]
    # labels_hc = labels_hc[:patient_min_length]
    #
    # features_scz = features_scz[:patient_min_length]
    # labels_scz = labels_scz[:patient_min_length]

    features_ = np.concatenate((features_hc, features_scz), axis=0)
    labels_ = np.concatenate((labels_hc, labels_scz), axis=0)

    kf = StratifiedKFold(n_splits=5, shuffle=True)
    count = 0
    loss_function = torch.nn.BCEWithLogitsLoss()
    for train_idx, test_idx in kf.split(features_, labels_):
        count += 1
        train_x, train_y, test_x, test_y = features_[train_idx], labels_[train_idx], \
                                           features_[test_idx], labels_[test_idx]

        train_x = train_x.reshape(len(train_x), -1)
        test_x = test_x.reshape(len(test_x), -1)
        labels = np.concatenate((labels, test_y.reshape(-1, 1)), axis=0)
        edge_index = torch.LongTensor(np.array(edge_index)).to(device)
        model = Model(frequency_feature_size // 16,
                      time_feature_size // 16, demographic_size, hidden_1, 1, channel, device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        model.train()
        min_max_scaler = preprocessing.StandardScaler()
        train_x = min_max_scaler.fit_transform(train_x)
        test_x = min_max_scaler.transform(test_x)
        train_frequency = train_x[:, :frequency_feature_size].reshape(-1, 16, frequency_feature_size // 16)
        train_time_features = train_x[:, frequency_feature_size:frequency_feature_size + time_feature_size].reshape(-1,
                                                                                                                    16,
                                                                                                                    time_feature_size // 16)
        train_demographics = train_x[:, -2:]
        train_frequency = torch.FloatTensor(train_frequency).to(device)
        train_time_features = torch.FloatTensor(train_time_features).to(device)
        train_demographics = torch.FloatTensor(train_demographics).to(device)
        train_y = torch.FloatTensor(train_y).to(device)

        for epoch in range(10):
            loss_train = 0.
            for train_index in range(len(train_frequency)):
                train_pred_prob, train_pred_prob_ = model(train_frequency[train_index],
                                                          train_time_features[train_index],
                                                          train_demographics[train_index], edge_index)
                loss = loss_function(train_pred_prob, torch.FloatTensor(train_y[train_index].reshape(-1, 1)).to(device))

            loss_train += loss
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

        model.eval()
        test_frequency = test_x[:, :frequency_feature_size].reshape(-1, 16, frequency_feature_size // 16)
        test_time_features = test_x[:, frequency_feature_size:frequency_feature_size + time_feature_size].reshape(-1,
                                                                                                                  16,
                                                                                                                  time_feature_size // 16)
        test_demographics = test_x[:, -2:]

        test_frequency = torch.FloatTensor(test_frequency).to(device)
        test_time_features = torch.FloatTensor(test_time_features).to(device)
        test_demographics = torch.FloatTensor(test_demographics).to(device)

        predicted_y_prob_all = torch.FloatTensor(np.zeros(shape=(0, 1))).to(device)
        for test_index in range(len(test_frequency)):
            predicted_y_prob, predicted_y_prob_ = model(test_frequency[test_index],
                                                        test_time_features[test_index],
                                                        test_demographics[test_index],
                                                        edge_index)
            predicted_y_prob_all = torch.cat((predicted_y_prob_all, predicted_y_prob_), dim=0)

        test_loss = loss_function(predicted_y_prob_all, torch.FloatTensor(test_y.reshape(-1, 1)).to(device))
        predicted_y_prob_all = predicted_y_prob_all.detach().cpu().numpy()
        acc, recall, precision, f1, specificity, auc = evaluate_model_performance_two_class(test_y,
                                                                                            predicted_y_prob_all,
                                                                                            flag=False)
        predicted_labels_prob = np.concatenate((predicted_labels_prob, predicted_y_prob_all), axis=0)  # 5折交叉结果保存
        print("validation {:01d} | train_loss(s) {:.4f} | test_loss {:.4f} | Acc {:.4f} | "
              "recall {:.4f} | precision {:.4f} | F1 {:.4f} | specificity {:.4f} | AUC {:.4f}"
              .format(count, loss_train.detach().numpy(), test_loss, acc, recall, precision, f1, specificity,
                      auc))

    acc, recall, precision, f1, specificity, auc = evaluate_model_performance_two_class(labels,
                                                                                        predicted_labels_prob,
                                                                                        flag=True)
    return acc, recall, precision, f1, specificity, auc


def optimize_parameters(learning_rate, hidden_1):
    learning_rate = 10 ** learning_rate
    hidden_1 = 2 ** int(hidden_1)
    print(f'learning_rate: {learning_rate}, hidden_size: {hidden_1}')
    all_acc = []
    all_recall = []
    all_precision = []
    all_f1 = []
    all_specificity = []
    all_auc = []
    for i in range(10):
        print(i, end=',')
        acc, recall, precision, f1, specificity, auc = run_experiment(-1, learning_rate, hidden_1)
        all_acc.append(acc)
        all_recall.append(recall)
        all_precision.append(precision)
        all_f1.append(f1)
        all_specificity.append(specificity)
        all_auc.append(auc)

    print('final_result: auc--{:.3f}±{:.3f}-acc--{:.3f}±{:.3f}--precision--{:.3f}±{:.3f}-'
          '-recall--{:.3f}±{:.3f}--f1--{:.3f}±{:.3f}-'
          '-specificity--{:.3f}±{:.3f}'.format(np.mean(all_auc),
                                               np.std(all_auc),
                                               np.mean(all_acc),
                                               np.std(all_acc),
                                               np.mean(all_precision),
                                               np.std(all_precision),
                                               np.mean(all_recall),
                                               np.std(all_recall),
                                               np.mean(all_f1),
                                               np.std(all_f1),
                                               np.mean(all_specificity),
                                               np.std(all_specificity)))
    return np.mean(all_auc)


if __name__ == '__main__':
    save_result('GAT-华西.txt')
    train_rnn_BO = BayesianOptimization(
        optimize_parameters, {
            'learning_rate': (-5, 0),
            'hidden_1': (4, 8),
        }
    )
    train_rnn_BO.maximize()
    print(train_rnn_BO.max)