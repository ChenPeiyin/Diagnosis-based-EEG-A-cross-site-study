import numpy as np
import torch
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
import logging
from torch.utils.data import Dataset
from typing import Optional, Sequence
from torch import Tensor
from torch.nn import functional as F
import torch.nn as nn
from torch.autograd import Variable
import copy
from torch.autograd import Function

def save_logging(file_name):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 设置打印级别
    formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s')

    # 设置屏幕打印的格式
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # 设置log保存
    fh = logging.FileHandler(file_name, encoding='utf8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logging.info('Start print log......')


def load_data(path, index, balance=False):
    features_ = np.load(path + '/all_patient_frequency_features_15360.npy').reshape(-1, 16, 5)
    features_de = np.load(path + '/all_patient_de_features_15360.npy').reshape(-1, 16, 5)
    features_dasm = np.load(path + '/all_patient_dasm_features_15360.npy').reshape(-1, 8, 5)
    features_rasm = np.load(path + '/all_patient_rasm_features_15360.npy').reshape(-1, 8, 5)
    features_dcau = np.load(path + '/all_patient_dcau_features_15360.npy').reshape(-1, 6, 5)
    features_ = np.concatenate((features_, features_de,
                                features_dasm, features_rasm,
                                features_dcau
                                ), axis=1)
    if index == -1:
        features_ = features_[:, :, :]
    else:
        features_ = features_[:, :, index]

    labels_ = np.load(path + '/all_patient_frequency_labels_15360.npy').reshape(-1, )
    age_and_sex_features = np.load(path + '/all_patient_age_and_sex.npy').reshape(-1, 2)
    time_mean_and_std = np.load(path + '/all_patient_time_features.npy').reshape(-1, 16, 14)

    hc_index = 2
    scz_index = 3
    labels_hc_index = np.argwhere(labels_ == hc_index)
    features_hc = features_[labels_hc_index]
    age_and_sex_features_hc = age_and_sex_features[labels_hc_index]
    time_mean_and_std_hc = time_mean_and_std[labels_hc_index]
    features_hc = np.concatenate(
        (features_hc.reshape(len(features_hc), -1),
         age_and_sex_features_hc.reshape(len(age_and_sex_features_hc), -1),
         time_mean_and_std_hc.reshape(len(time_mean_and_std_hc), -1)
         ),
        axis=1)
    labels_hc = np.zeros(shape=[len(labels_hc_index)])

    labels_scz_index = np.argwhere(labels_ == scz_index)
    features_scz = features_[labels_scz_index]
    age_and_sex_features_scz = age_and_sex_features[labels_scz_index]
    time_mean_and_std_scz = time_mean_and_std[labels_scz_index]
    features_scz = np.concatenate((features_scz.reshape(len(features_scz), -1),
                                   age_and_sex_features_scz.reshape(len(age_and_sex_features_scz), -1),
                                   time_mean_and_std_scz.reshape(len(time_mean_and_std_scz), -1)
                                   ), axis=1)
    labels_scz = np.ones(shape=[len(labels_scz_index)])

    features_ = np.concatenate((features_hc, features_scz), axis=0)
    labels_ = np.concatenate((labels_hc, labels_scz), axis=0)
    labels_ = labels_.astype(np.int64)
    if balance:
        features_, labels_ = SMOTE().fit_resample(features_, labels_)
    return shuffle(features_, labels_)


def shuffle(features, labels):
    index = np.arange(len(features))
    np.random.shuffle(index)
    features = features[index]
    labels = labels[index]
    return features, labels

class EEGDataset(Dataset):
    def __init__(self, X, y):
        X = X.astype('float')
        # y = y.astype('int')
        dataSize = X.shape
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.len = len(y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len

class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = x
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

class GradientReverseLayer(Function):
    @staticmethod
    def forward(ctx, x, theta):
        ctx.theta = theta

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.theta

        return output, None

class DANet(nn.Module):
    def __init__(self, backbone, frequency_feature_size, time_feature_size, fea_dim, hidden_size, n_class=2):
        super(DANet, self).__init__()
        self.__dict__.update(locals())
        del self.self
        self.frequency_feature_size = frequency_feature_size
        self.time_feature_size = time_feature_size
        self.fea_dim = fea_dim

        self.Fea = copy.deepcopy(backbone)

        self.Classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(fea_dim, 2),  # hidden_size
            nn.LogSoftmax(1)
        )

        self.Discrinminator = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(fea_dim, 2),  # hidden_size
            nn.LogSoftmax(1)
        )

    def forward(self, source_data, target_data, theta=1):
        # print(source_data.size(), target_data.size())
        s_fea, _, _ = self.Fea(source_data[:, :self.frequency_feature_size].reshape(-1, 16, 10),
                               source_data[:,
                               self.frequency_feature_size:self.frequency_feature_size + self.time_feature_size].reshape(-1,
                                                                                                                    16,
                                                                                                                    14),
                               source_data[:, self.frequency_feature_size + self.time_feature_size:])
        # print(s_fea.size())

        s_clf = self.Classifier(s_fea)
        source_feature = s_fea.view(s_fea.shape[0], -1)
        # print(source_feature.size())

        t_fea, _, _ = self.Fea(target_data[:, :self.frequency_feature_size].reshape(-1, 16, 10),
                               target_data[:,
                               self.frequency_feature_size:self.frequency_feature_size + self.time_feature_size].reshape(-1,16,14),
                               target_data[:, self.frequency_feature_size + self.time_feature_size:])
        # print(t_fea.size())

        t_clf = self.Classifier(t_fea)
        target_feature = t_fea.view(t_fea.shape[0], -1)

        if self.training:
            re_source_feature = GradientReverseLayer.apply(source_feature, theta)
            re_target_feature = GradientReverseLayer.apply(target_feature, theta)

            s_output = self.Discrinminator(re_source_feature)
            t_output = self.Discrinminator(re_target_feature)

        else:
            s_output = self.Discrinminator(s_fea)
            t_output = self.Discrinminator(t_fea)

        return source_feature, target_feature, s_clf, t_clf, s_output, t_output

