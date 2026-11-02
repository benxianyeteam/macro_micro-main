import torch
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from OT_torch_ import cost_matrix_batch_torch, GW_distance_uniform, IPOT_distance_torch_batch_uniform

class MMD_loss(torch.nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta)
        return loss

    def forward(self, source, target): # 输入是16*384和16*384
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
                del XX, YY, XY, YX
            torch.cuda.empty_cache()
            return loss

def label2edge(label, device):
    """
    convert ground truth labels into ground truth edges
    :param label: ground truth labels
    :param device: the gpu device that holds the ground truth edges
    :return: ground truth edges
    """
    # get size
    num_samples = label.size(0)
    # reshape
    label_i = label.unsqueeze(-1).repeat(1, num_samples)
    label_j = label_i.transpose(1, 0)
    # compute edge
    edge = torch.eq(label_i, label_j).float().to(device)
    return edge


# initialize nodes of distribution graph
def initialize_nodes_edges(data, label, num_class):
    """
    :param batch: data batch
    :param num_supports: number of samples in support set
    :param tensors: initialized tensors for holding data
    :param batch_size: how many tasks per batch
    :param num_queries: number of samples in query set
    :param num_ways: number of classes for each few-shot task
    :param device: the gpu device that holds all data

    :return: data of support set,
             label of support set,
             data of query set,
             label of query set,
             data of support and query set,
             label of support and query set,
             initialized node features of distribution graph (Vd_(0)),
             initialized edge features of point graph (Ep_(0)),
             initialized edge_features_of distribution graph (Ed_(0))
    """
    # initialize nodes of distribution graph
    device = data.device
    label_one_hot = F.one_hot(label, num_classes=num_class)  # n为类别数
    distribution_graph = label2edge(label, device)

    # initialize edges of point graph
    # 由于 cosine_similarity 期望输入是一个形状为 (n_samples, n_features) 的二维数组
    # 我们需要将每个样本平展（flatten）成一个向量
    data_flattened = data.reshape(data.shape[0], -1)

    # 使用 cosine_similarity 计算所有样本对之间的余弦相似度
    # 结果是一个 10x10 的矩阵

    # 将Tensor转换为NumPy数组（需要先分离梯度）
    data_flattened_np = data_flattened.detach().cpu().numpy()
    cosine_similarities = cosine_similarity(data_flattened_np)
    similarity_matrix = torch.tensor(cosine_similarities)

    # initialize edges of distribution graph (same as point graph)
    feature_graph = similarity_matrix.to(device).clone()

    return label_one_hot, distribution_graph, feature_graph


def OT(src, tar, ori=False, sub=False, **kwargs):
    wd, gwd = [], []
    src, tar = src.unsqueeze(0), tar.unsqueeze(0)
    source_share, target_share = src, tar
    cos_distance = cost_matrix_batch_torch(source_share, target_share)
    cos_distance = cos_distance.transpose(1, 2)
    # TODO: GW as graph matching loss
    beta = 0.1
    if sub:
        cos_distance = kwargs['w_st'] * cos_distance
    min_score = cos_distance.min()
    max_score = cos_distance.max()
    threshold = min_score + beta * (max_score - min_score)
    cos_dist = torch.nn.functional.relu(cos_distance - threshold)

    wd_val = - IPOT_distance_torch_batch_uniform(cos_dist, source_share.size(0), source_share.size(2),
                                                 target_share.size(2), iteration=30)
    gwd_val = GW_distance_uniform(source_share, target_share, sub, **kwargs)
    wd.append(abs(wd_val))
    gwd.append(abs(gwd_val))

    ot = sum(wd) / len(wd) + sum(gwd) / len(gwd)
    return ot, sum(wd) / len(wd), sum(gwd) / len(gwd)

