import copy
from torch.nn import Conv2d, BatchNorm2d, PReLU, MaxPool2d, Sequential
from collections import namedtuple
from models.GraphConvolutionNetwork import *

class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''

def get_block(in_channel, depth, num_units, stride=2):

    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]

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

class bottleneck_IR(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut

def get_blocks(num_layers):
    if num_layers == 10:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=1),
            get_block(in_channel=64, depth=128, num_units=1),
            get_block(in_channel=128, depth=256, num_units=1),
            get_block(in_channel=256, depth=512, num_units=1)
        ]
    if num_layers == 18:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=2),
            get_block(in_channel=64, depth=128, num_units=2),
            get_block(in_channel=128, depth=256, num_units=2),
            get_block(in_channel=256, depth=512, num_units=2)
        ]
    elif num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]

    return blocks


class Transfer_Network(nn.Module):
    def __init__(self, numOfLayer, num_class):
        super(Transfer_Network, self).__init__()

        unit_module = bottleneck_IR

        self.num_class = num_class

        self.input_layer = Sequential(
            Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            BatchNorm2d(64),
            PReLU(64))

        blocks = get_blocks(numOfLayer)

        self.layer1 = Sequential(*[unit_module(bottleneck.in_channel, bottleneck.depth, bottleneck.stride) for bottleneck in blocks[0]])  # get_block(in_channel=64, depth=64, num_units=3)])
        self.layer2 = Sequential(*[unit_module(bottleneck.in_channel, bottleneck.depth, bottleneck.stride) for bottleneck in blocks[1]])  # get_block(in_channel=64, depth=128, num_units=4)])
        self.layer3 = Sequential(*[unit_module(bottleneck.in_channel, bottleneck.depth, bottleneck.stride) for bottleneck in blocks[2]])  # get_block(in_channel=128, depth=256, num_units=14)])
        self.layer4 = Sequential(*[unit_module(bottleneck.in_channel, bottleneck.depth, bottleneck.stride) for bottleneck in blocks[3]])  # get_block(in_channel=256, depth=512, num_units=3)])

        self.output_layer = Sequential(
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))

        cropNet_modules = []
        cropNet_blocks = [get_block(in_channel=128, depth=256, num_units=2),
                          get_block(in_channel=256, depth=512, num_units=2)]
        for block in cropNet_blocks:
            for bottleneck in block:
                cropNet_modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        cropNet_modules += [
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.ReLU()]
        self.Crop_Net = nn.ModuleList([copy.deepcopy(nn.Sequential(*cropNet_modules)) for i in range(5)])

        self.fc = nn.Linear(64 + 320, self.num_class)
        # self.fc = nn.Linear(64, 3)
        self.fc.apply(init_weights)

        self.loc_fc = nn.Linear(320, self.num_class)
        self.loc_fc.apply(init_weights)

        self.GAP = nn.AdaptiveAvgPool2d((1, 1))

        self.TransferModule = Transfer_Attention_Module(dim_in=64, dim_hid=128, dim_out=64)

    def transfer(self, x, locations, transfer_matrix):
        featureMap = self.input_layer(x)  # Batch * 64 * 112 * 112

        featureMap1 = self.layer1(featureMap)  # Batch * 64 * 56 * 56
        featureMap2 = self.layer2(featureMap1)  # Batch * 128 * 28 * 28
        featureMap3 = self.layer3(featureMap2)  # Batch * 256 * 14 * 14
        featureMap4 = self.layer4(featureMap3)  # Batch * 512 * 7 * 7

        global_feature = self.output_layer(featureMap4).view(featureMap.size(0), -1)  # Batch * 64
        # feature = global_feature  # Batch * (64+320)
        loc_feature = self.crop_featureMap(featureMap2, locations)  # Batch * 320
        feature = torch.cat((global_feature, loc_feature), 1)  # Batch * (64+320)

        # Compute Feature
        macro_feature = feature.narrow(0, 0, feature.size(0) // 2)  # Batch/2 * (64+320)
        macro_feature = macro_feature.view(macro_feature.size(0), 6, -1)

        micro_feature = feature.narrow(0, feature.size(0) // 2, feature.size(0) // 2)  # Batch/2 * (64+320)
        micro_feature = micro_feature.view(micro_feature.size(0), 6, -1)

        macro_feature_new, micro_feature_new = self.TransferModule(macro_feature, micro_feature, transfer_matrix)
        feature_end = torch.cat((macro_feature_new, micro_feature_new), dim=0)
        y = self.fc(feature_end)
        return feature_end, y

    def classify(self, x, locations):
        featureMap = self.input_layer(x)  # Batch * 64 * 112 * 112

        featureMap1 = self.layer1(featureMap)  # Batch * 64 * 56 * 56
        featureMap2 = self.layer2(featureMap1)  # Batch * 128 * 28 * 28
        featureMap3 = self.layer3(featureMap2)  # Batch * 256 * 14 * 14
        featureMap4 = self.layer4(featureMap3)  # Batch * 512 * 7 * 7

        global_feature = self.output_layer(featureMap4).view(featureMap.size(0), -1)  # Batch * 64
        # feature = global_feature  # Batch * (64+320)
        loc_feature = self.crop_featureMap(featureMap2, locations)  # Batch * 320
        feature_end = torch.cat((global_feature, loc_feature), 1)  # Batch * (64+320)

        y = self.fc(feature_end)
        return feature_end, y

    def forward(self, x, locations=None, transfer_matrix=None, flag=None):

        if flag == True and transfer_matrix is not None:
            return self.transfer(x, locations, transfer_matrix)
        else:
            return self.classify(x, locations)

    def crop_featureMap(self, featureMap, locations):
        batch_size = featureMap.size(0)
        map_ch = featureMap.size(1)
        map_len = featureMap.size(2)

        grid_ch = map_ch
        grid_len = 7  # 14, 6, 4

        feature_list = []
        for i in range(5):
            grid_list = []
            for j in range(batch_size):
                w_min = locations[j, i, 0] - int(grid_len / 2)
                w_max = locations[j, i, 0] + int(grid_len / 2)
                h_min = locations[j, i, 1] - int(grid_len / 2)
                h_max = locations[j, i, 1] + int(grid_len / 2)

                map_w_min = max(0, w_min)
                map_w_max = min(map_len - 1, w_max)
                map_h_min = max(0, h_min)
                map_h_max = min(map_len - 1, h_max)

                grid_w_min = max(0, 0 - w_min)
                grid_w_max = grid_len + min(0, map_len - 1 - w_max)
                grid_h_min = max(0, 0 - h_min)
                grid_h_max = grid_len + min(0, map_len - 1 - h_max)

                grid = torch.zeros(grid_ch, grid_len, grid_len)

                if featureMap.is_cuda:
                    # 获取当前数据所在的 GPU 设备
                    device_idx = featureMap.get_device()
                    grid = grid.cuda(device_idx)

                grid[:, grid_h_min:grid_h_max + 1, grid_w_min:grid_w_max + 1] = featureMap[j, :,
                                                                                map_h_min:map_h_max + 1,
                                                                                map_w_min:map_w_max + 1]

                grid_list.append(grid)

            feature = torch.stack(grid_list, dim=0)
            feature_list.append(feature)

        # feature list: 5 * [ batch_size * channel * 3 * 3 ]
        output_list = []
        for i in range(5):
            output = self.Crop_Net[i](feature_list[i])
            output = self.GAP(output)
            output_list.append(output)

        loc_feature = torch.stack(output_list, dim=1)  # batch_size * 5 * 64 * 1 * 1
        loc_feature = loc_feature.view(batch_size, -1)  # batch_size * 320

        return loc_feature

