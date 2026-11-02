import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class GraphConvolution(nn.Module):
    """
    Document : https://github.com/tkipf/pygcn
               https://github.com/bamos/block
    """
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.FC = nn.Linear(in_features, out_features)
        self.FC.apply(init_weights)

    def forward(self, input, adj):
        batchSize, numOfNode = input.size(0), adj.size(0)
        support = self.FC(input)
        Adj = adj.repeat(batchSize, 1, 1)
        output = torch.bmm(Adj, support)

        return output

class TGCN(nn.Module):
    def __init__(self, in_features, out_features, ):
        super(TGCN, self).__init__()
        self.gamma = 0.1

        self.in_features = in_features * 6

        self.fc = nn.Linear(self.in_features, self.in_features)
        self.fc.apply(init_weights)

    def forward(self, Z, T):
        # 确保T是适合批处理的维度
        assert T.size(0) == Z.size(0) and T.size(1) == Z.size(0), "T should have shape [batch_size, batch_size]"
        Z = Z.view(Z.size(0), -1)  # 形状变为 [48, 384]
        Z = self.fc(Z)

        # 线性变换
        support = torch.matmul(Z,
                               self.fc.weight)  # Z: [batch_size, num_nodes, in_features], support: [batch_size, num_nodes, out_features]
        Z_transformed = torch.matmul(T, support)  # [batch_size, num_nodes, out_features]
        # 应用激活函数
        Z_updated = self.gamma * F.relu(Z_transformed)

        return Z_updated

class RGCN(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out):
        super(RGCN, self).__init__()
        # 定义两个图卷积层
        self.gcn1 = GraphConvolution(dim_in, dim_hid)
        self.gcn2 = GraphConvolution(dim_hid, dim_out)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_data, adj_matrix):
        # 首先应用第一个图卷积层
        x = self.gcn1(input_data, adj_matrix)

        x = self.dropout(x)
        x = self.relu(x)
        # 然后应用第二个图卷积层
        x = self.gcn2(x, adj_matrix)

        return x


class Transfer_Attention_Module(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out, beta_s=1.0, beta_t=1.0):
        super(Transfer_Attention_Module, self).__init__()
        # 这里将原来的MLP替换成GCN来做映射
        self.softmax = nn.Softmax(-1)
        self.beta_s = beta_s
        self.beta_t = beta_t
        self.scale = 0.1

        self.start_relation = RGCN(dim_in, dim_hid, dim_out)

        self.pre_project = TGCN(dim_in, dim_hid)

        self.post_project = nn.Sequential(  # only one layer
            nn.Linear(dim_hid, dim_out * 6)
        )
    def forward(self, macro_feature, micro_feature, transfer_matrix):
        '''
        Fs: 源域数据
        Ft: 目标域数据
        Fs_bank: 源域的类中心
        Ft_bank: 目标域的类中心
        transfer_matrix: 转换矩阵，当源域和目标域的标签一样的时候取值为1，不一样的时候取值为0
        '''
        node_adjacent_matrix_macro = self.softmax(self.get_adjacent_matrix(macro_feature)) # 采用softmax的形式对邻接矩阵归一化
        node_adjacent_matrix_micro = self.softmax(self.get_adjacent_matrix(micro_feature))  # 采用softmax的形式对邻接矩阵归一化
        #
        out_macro = self.start_relation(macro_feature, node_adjacent_matrix_macro) # 考虑了节点相关性的源域
        out_micro = self.start_relation(micro_feature, node_adjacent_matrix_micro) # 考虑了节点相关性的目标域

        out_fs2 = macro_feature.view(macro_feature.size(0), -1) + self.scale * self.pre_project(out_micro, transfer_matrix) # 考虑了迁移矩阵的源域
        out_ft2 = micro_feature.view(micro_feature.size(0), -1) + self.scale * self.pre_project(out_macro, transfer_matrix.t()) # 考虑了迁移矩阵的目标域

        # q_fs, k_fs, v_fs = tuple(rearrange(out_fs2, 'b (d k) -> k b d ', k=3))
        # q_ft, k_ft, v_ft = tuple(rearrange(out_ft2, 'b (d k) -> k b d ', k=3))
        # # out_fs2 = macro_feature.view(macro_feature.size(0), -1)
        # # out_ft2 = micro_feature.view(micro_feature.size(0), -1)
        # # q_fs, k_fs, v_fs = tuple(rearrange(out_fs2, 'b (d k) -> k b d ', k=3))
        # # q_ft, k_ft, v_ft = tuple(rearrange(out_ft2, 'b (d k) -> k b d ', k=3))
        #
        # q_f = 0.1 * q_fs + 0.1 * q_ft
        # # #从这里出现nan
        # As = self.softmax(q_f @ k_fs.permute(1, 0))  # (batch, N)
        # At = self.softmax(q_f @ k_ft.permute(1, 0))  # (batch, N)
        #
        # Fsa = out_fs2 + self.post_project(As @ v_fs)  # (batch, C)
        # Fta = out_ft2 + self.post_project(At @ v_ft)  # (batch, C)
        # #
        # Fsa = Fsa / Fsa.norm(dim=-1, keepdim=True)
        # Fta = Fta / Fta.norm(dim=-1, keepdim=True)

        return out_fs2, out_ft2

    # 得到节点的邻接矩阵
    def get_adjacent_matrix(self, micro_feature):
        center = micro_feature.transpose(0, 1)  # 将第一个维度和第二个维度进行交换
        # 获取当前数据所在的 GPU 设备
        device_idx = center.get_device()
        # 初始化一个6x6的零矩阵来存储相似度
        similarity_matrix = torch.zeros(center.shape[0], center.shape[0]).cuda(device_idx)
        for i in range(center.shape[0]):
            for j in range(center.shape[0]):
                pairwiseDistance = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)
                similarity_matrix[i, j] = pairwiseDistance(center[i, :, :].reshape(-1), center[j, :, :].reshape(-1))
        # print(similarity_matrix)
        return similarity_matrix