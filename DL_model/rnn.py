import torch.nn as nn
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from evaluate_model import evaluate_model_performance_two_class
from bayes_opt import BayesianOptimization
from save_results import save_result
import warnings

warnings.filterwarnings('ignore')


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(input_size, hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        # forward propagate lstm
        input = torch.permute(input, [0, 2, 1])
        out, (h_n, h_c) = self.lstm(input[:, :15360, :], None)
        out = self.fc(out[:, -1, :])
        output = torch.sigmoid(out)
        return output


def train_rnn(learning_rate, l2_regularization):
    hidden_size = 32
    learning_rate = 10 ** learning_rate
    l2_regularization = 10 ** l2_regularization

    print('learning_rate--{}--l2_regularization--{}--hidden_size--{}'.format(learning_rate, l2_regularization,
                                                                             hidden_size))
    # features_ = np.load('all_patient_frequency_features_multi_psd.npy', allow_pickle=True)[:, :, :, 4]
    # labels_ = np.load('all_patient_frequency_labels_multi_psd.npy', allow_pickle=True).reshape(-1, )

    features_ = np.load('../../extracted_dataset/huaxi/no_sampling/all_patient_features.npy', allow_pickle=True)[:, :, :1280]
    labels_ = np.load('../../extracted_dataset/huaxi/no_sampling/all_patient_labels.npy', allow_pickle=True).reshape(-1, )

    features_ = features_.reshape(len(features_), features_.shape[1], -1)
    # features_ = np.swapaxes(features_, 1, 2)  # 频域需要交换！！！
    hc_index = 1
    scz_index = 2
    labels_hc_index = np.argwhere(labels_ == hc_index)
    features_hc = features_[labels_hc_index]
    features_hc = features_hc.reshape(len(labels_hc_index), features_hc.shape[-2], features_hc.shape[-1])
    labels_hc = np.zeros(shape=[len(labels_hc_index)])

    labels_scz_index = np.argwhere(labels_ == scz_index)
    features_scz = features_[labels_scz_index]
    features_scz = features_scz.reshape(len(labels_scz_index), features_scz.shape[-2], features_scz.shape[-1])
    labels_scz = np.ones(shape=[len(labels_scz_index)])

    features_ = np.concatenate((features_hc, features_scz), axis=0)
    labels_ = np.concatenate((labels_hc, labels_scz), axis=0)

    input_size = features_.shape[1]
    output_size = 1
    epochs = 20
    batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predicted_labels_prob = np.zeros(shape=[0, output_size])
    predicted_labels = np.zeros(shape=[0, 1])
    labels = np.zeros(shape=[0, 1])

    kf = StratifiedKFold(n_splits=5, shuffle=False)
    cv = -1
    for train_idx, test_idx in kf.split(features_, labels_):
        cv += 1
        model = RNN(input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size)
        model.to(device=device)

        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

        train_x, train_y, test_x, test_y = features_[train_idx], labels_[train_idx], features_[test_idx], labels_[test_idx]
        # scaler = MinMaxScaler()
        # scaler.fit(train_x.reshape(len(train_x), -1))
        # train_x = scaler.transform(train_x.reshape(len(train_x), -1))
        # train_x = train_x.reshape(-1, input_size, features_.shape[2])
        # test_x = scaler.transform(test_x.reshape(len(test_x), -1))
        # test_x = test_x.reshape(-1, input_size, features_.shape[2])
        train_x, train_y = shuffle(train_x, train_y)
        test_x, test_y = shuffle(test_x, test_y)

        loss_function = nn.BCELoss()
        label = torch.FloatTensor(train_y).to(device)

        for epoch in range(epochs):
            # training process
            output = torch.zeros(size=[0, output_size]).to(device)
            for i in range(int(len(train_x)/batch_size)+1):
                if (i+1) * batch_size < len(train_x):
                    train_x_batch = train_x[i*batch_size:(i+1)*batch_size, :, :]
                else:
                    train_x_batch = train_x[i*batch_size:, :, :]
                output_ = model(torch.FloatTensor(train_x_batch).to(device))
                output = torch.cat((output, output_), dim=0)

            loss_bce = loss_function(output.reshape(-1,), label.reshape(-1,))
            optimizer.zero_grad()
            loss_bce.backward(retain_graph=True)
            optimizer.step()

            train_auc, train_acc, train_precision, train_recall, train_f1, train_specificity = \
                get_model_performance(output.detach().cpu().numpy(), train_y, False)

            # testing process
            output = torch.zeros(size=[0, output_size]).to(device)
            for i in range(int(len(test_x) / batch_size) + 1):
                if (i + 1) * batch_size < len(test_x):
                    test_x_batch = test_x[i * batch_size:(i + 1) * batch_size, :, :]
                else:
                    test_x_batch = test_x[i * batch_size:, :, :]
                output_ = model(torch.FloatTensor(test_x_batch).to(device))
                output = torch.cat((output, output_), dim=0)

            test_auc, test_acc, test_precision, test_recall, test_f1, test_specificity = get_model_performance(output.detach().cpu().numpy(),
                                                                                                               test_y,
                                                                                                               False)
            predicted_label = torch.argmax(output, dim=-1)
            print('cross_validation--{}-epoch-{}-train_auc--{:.3f}--train_acc--{:.3f}--'
                  'train_precision--{:.3f}--train_recall--{:.3f}--'
                  'train_f1-{:.3f}--train_specificity{:.3f}--test_auc--{:.3f}--test_acc--{:.3f}--test_precision-'
                  '-{:.3f}--test_recall {:.3f}--test_f1--{:.3f}--test_specificity--{:.3f}'.
                  format(cv, epoch, train_auc, train_acc, train_precision, train_recall, train_f1, train_specificity,
                         test_auc, test_acc, test_precision, test_recall, test_f1, test_specificity))

            predicted_labels_prob = np.concatenate((predicted_labels_prob, output.detach().cpu().numpy()), axis=0)
            predicted_labels = np.concatenate((predicted_labels, predicted_label.detach().cpu().numpy().reshape(-1, 1)),
                                              axis=0)
            labels = np.concatenate((labels, test_y.reshape(-1, 1)), axis=0)

    acc, recall, precision, f1, specificity, auc = evaluate_model_performance_two_class(labels,
                                                                                        predicted_labels_prob)
    return acc
    # return acc, recall, precision, f1, specificity, auc


def get_model_performance(output, label, flag=True):
    acc, recall, precision, f1, specificity, auc = evaluate_model_performance_two_class(label,
                                                                                        output,
                                                                                        flag)
    return auc, acc, precision, recall, f1, specificity


def shuffle(features, labels):
    index = np.arange(len(features))
    np.random.shuffle(index)
    features = features[index]
    labels = labels[index]
    return features, labels


if __name__ == '__main__':
    save_result('results_rnn_时域特征_正常+抑郁_1280.txt')
    train_rnn_BO = BayesianOptimization(
            train_rnn, {
                'learning_rate': (-5, 0),
                'l2_regularization': (-5, 1),
            }
        )
    train_rnn_BO.maximize()
    print(train_rnn_BO.max)

    # macro_aucs, accuracys, macro_precisions, macro_recalls, macro_f1s = [[] for i in range(5)]
    # for i in range(10):
    #     macro_auc, accuracy, macro_precision, macro_recall, macro_f1 = train_rnn(learning_rate=0.00132290113658427,
    #                                                                              l2_regularization=0.1366752274263632)
    #     macro_aucs.append(macro_auc)
    #     accuracys.append(accuracy)
    #     macro_precisions.append(macro_precision)
    #     macro_recalls.append(macro_recall)
    #     macro_f1s.append(macro_f1)
    #
    # print('auc--{}--accuracy--{}--precision--{}--recall--{}--f1--{}'.format(np.mean(macro_aucs),
    #                                                                         np.mean(accuracys),
    #                                                                         np.mean(macro_precisions),
    #                                                                         np.mean(macro_recalls),
    #                                                                         np.mean(macro_f1s)))
