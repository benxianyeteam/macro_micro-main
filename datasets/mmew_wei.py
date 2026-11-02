import os
from PIL import Image
import numpy as np
import torch
import random
from torch.utils.data import Dataset
import pandas as pd
from glob import glob

emotion_to_label3 = {"disgust": 0,
                    "fear": 0,
                    "sadness": 0,
                    "anger":0,
                    "happiness": 1,
                    "surprise": 2,
                    }

emotion_to_label_5 = {"happiness": 0,
                    "disgust": 1,
                    "fear": 2,
                    "sadness": 3,
                    "surprise": 4,
                    }

emotion_to_label_6 = {"happiness": 0,
                    "disgust": 1,
                    "fear": 2,
                    "sadness": 3,
                    "surprise": 4,
                    "anger": 5
                    }

class MMEW_wei_six(Dataset):

    def __init__(self, data_name, transform=None, transform_norm=None, **kwargs,):

        self.split = kwargs['split']
        self.data_name = data_name
        df = pd.read_excel(kwargs['excel'])
        # 将表格中的1-26修改为sub01-sub26，方便后面使用
        df['Subject'] = df['Subject'].apply(str)
        df['Subject'].replace({'1': 'S01', '2': 'S02', '3': 'S03', '4': 'S04', '5': 'S05',
                               '6': 'S06', '7': 'S07', '8': 'S08', '9': 'S09', '10': 'S10',
                               '11': 'S11', '12': 'S12', '13': 'S13', '14': 'S14', '15': 'S15',
                               '16': 'S16', '17': 'S17', '18': 'S18', '19': 'S19', '20': 'S20',
                               '21': 'S21', '22': 'S22', '23': 'S23', '24': 'S24', '25': 'S25',
                               '26': 'S26', '27': 'S27', '28': 'S28', '29': 'S29', '30': 'S30'
                               }, inplace=True)

        root_rgb = kwargs['path_rgb']
        bbox_path = kwargs['bboxs']
        landmark_path = kwargs['landmakes']
        if kwargs['split'] == 'train': # 训练集
            dataset = df.loc[df['Subject'] != kwargs['sub_id']]
        else: # 测试集
            dataset = df.loc[df['Subject'] == kwargs['sub_id']]

        Subject = dataset['Subject'].values
        Filename = dataset['Filename'].values
        OnsetFrame = dataset['OnsetFrame'].values

        ApexFrame = dataset['ApexFrame'].values
        OffsetFrame = dataset['OffsetFrame'].values
        Action_Units = dataset['Action Units'].values
        Estimated_Emotion = dataset['Estimated Emotion'].values

        # 得到数据和标签
        self.data_rgb, self.label, self.domain_label, self.bboxs, self.landmarks = [], [], [], [], []
        self.onset, self.apexset, self.offset = [], [], []
        if self.data_name == 'mmew_wei':
            for i in range(len(Estimated_Emotion)):
                emotion = Estimated_Emotion[i]
                if emotion == 'happiness' or emotion == 'disgust' or emotion == 'fear' or emotion == 'sadness' or emotion == 'surprise' or emotion == 'anger':
                    self.data_rgb.append(os.path.join(root_rgb, Subject[i], Estimated_Emotion[i], Filename[i]))
                    self.label.append(emotion_to_label_6[emotion])
                    self.domain_label.append(int(1))
                    bboxs_txt_path = glob(os.path.join(bbox_path, Subject[i], Estimated_Emotion[i], Filename[i], '*'))
                    bbox = np.loadtxt(bboxs_txt_path[0])
                    self.bboxs.append(bbox)
                    landmarks_txt_path = glob(os.path.join(landmark_path, Subject[i], Estimated_Emotion[i], Filename[i], '*'))
                    landmark_5 = np.loadtxt(landmarks_txt_path[0])
                    self.landmarks.append(landmark_5)

                    self.onset.append(str(OnsetFrame[i])+'.jpg')
                    self.apexset.append(str(ApexFrame[i])+'.jpg')
                    self.offset.append(str(OffsetFrame[i])+'.jpg')

                    # 是否读取增强数据
                    if kwargs['split'] == 'train' and (emotion == 'fear' or emotion == 'sadness'):
                        root_rgb_aug = kwargs['path_rgb_aug']
                        aug_filename = os.path.join(root_rgb_aug, Subject[i], Estimated_Emotion[i], Filename[i])
                        aug_rgb_path = glob(os.path.join(aug_filename, '*'))
                        for h in aug_rgb_path:
                            self.data_rgb.append(h)
                            self.label.append(emotion_to_label_6[emotion])
                            self.domain_label.append(int(1))
                            self.bboxs.append(bbox)

                            self.onset.append('onset.jpg')
                            self.apexset.append('apex.jpg')
                            landmark_5 = np.load(os.path.join(h, 'apex_landmarks.npy'))

                            self.landmarks.append(landmark_5)

        self.num_class_list = np.unique(self.label)

        self.transform = transform
        self.transform_norm = transform_norm

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img_on, img_apex, label, domain_label, landmark = self.get_data_from_index(idx)
        return img_on, img_apex, label, domain_label, landmark

    def get_data_from_index(self, index):
        # /data/guoying/gyy/gyy_data/CK+/327labeled_CK+_cropped/happy/S037/
        '/data/guoying/gyy/gyy_data/CK+/327labeled_CK+_cropped/fear/S074/S074_001_00000020.png'
        path_onset = os.path.join(self.data_rgb[index], self.onset[index])
        path_apex = os.path.join(self.data_rgb[index], self.apexset[index])

        img_onset, img_apex = Image.open(path_onset).convert('RGB'), Image.open(path_apex).convert('RGB')

        label = self.label[index]
        domain_label = self.domain_label[index]
        bbox, landmark = self.bboxs[index], self.landmarks[index]

        ori_img_w, ori_img_h = img_onset.size

        # BoundingBox
        left = bbox[0]
        top = bbox[1]
        right = bbox[2]
        bottom = bbox[3]

        enlarge_bbox = True

        if self.data_name == 'mmew_wei':
            random_crop = True
        else:
            random_crop = False

        # Enlarge BoundingBox
        padding_w, padding_h = int(0.5 * max(0, int(0.20 * (right - left)))), int(
            0.5 * max(0, int(0.20 * (bottom - top))))

        if enlarge_bbox:
            left = max(left - padding_w, 0)
            right = min(right + padding_w, ori_img_w)

            top = max(top - padding_h, 0)
            bottom = min(bottom + padding_h, ori_img_h)

        if random_crop:
            x_offset = random.randint(-padding_w, padding_w)
            y_offset = random.randint(-padding_h, padding_h)

            left = max(left + x_offset, 0)
            right = min(right - x_offset, ori_img_w)

            top = max(top + y_offset, 0)
            bottom = min(bottom - y_offset, ori_img_h)

        img = img_onset.crop((left, top, right, bottom))
        crop_img_w, crop_img_h = img.size

        landmark[:, 0] -= left
        landmark[:, 1] -= top

        # Transform Image
        if self.transform is not None:
            img_onset = self.transform(img_onset)
            img_apex = self.transform(img_apex)

        ALL = torch.cat((img_onset, img_apex), dim=0)
        ## data augmentation for training only
        if self.transform_norm is not None and self.split == 'train':
            ALL = self.transform_norm(ALL)
        img_onset = ALL[0:3, :, :]
        img_apex = ALL[3:6, :, :]

        inputSizeOfCropNet = 28
        landmark[:, 0] = landmark[:, 0] * inputSizeOfCropNet / crop_img_w
        landmark[:, 1] = landmark[:, 1] * inputSizeOfCropNet / crop_img_h
        landmark = landmark.astype(int)

        grid_len = 7
        half_grid_len = int(grid_len / 2)

        for index in range(landmark.shape[0]):
            if landmark[index, 0] <= (half_grid_len - 1):
                landmark[index, 0] = half_grid_len
            if landmark[index, 0] >= (inputSizeOfCropNet - half_grid_len):
                landmark[index, 0] = inputSizeOfCropNet - half_grid_len - 1
            if landmark[index, 1] <= (half_grid_len - 1):
                landmark[index, 1] = half_grid_len
            if landmark[index, 1] >= (inputSizeOfCropNet - half_grid_len):
                landmark[index, 1] = inputSizeOfCropNet - half_grid_len - 1

        return img_onset, img_apex, label, domain_label, landmark

class MMEW_wei_five(Dataset):

    def __init__(self, data_name, transform=None, transform_norm=None, **kwargs,):

        self.split = kwargs['split']
        self.data_name = data_name
        df = pd.read_excel(kwargs['excel'])
        # 将表格中的1-26修改为sub01-sub26，方便后面使用
        df['Subject'] = df['Subject'].apply(str)
        df['Subject'].replace({'1': 'S01', '2': 'S02', '3': 'S03', '4': 'S04', '5': 'S05',
                               '6': 'S06', '7': 'S07', '8': 'S08', '9': 'S09', '10': 'S10',
                               '11': 'S11', '12': 'S12', '13': 'S13', '14': 'S14', '15': 'S15',
                               '16': 'S16', '17': 'S17', '18': 'S18', '19': 'S19', '20': 'S20',
                               '21': 'S21', '22': 'S22', '23': 'S23', '24': 'S24', '25': 'S25',
                               '26': 'S26', '27': 'S27', '28': 'S28', '29': 'S29', '30': 'S30'
                               }, inplace=True)

        root_rgb = kwargs['path_rgb']
        root_flow = kwargs['path_flow']
        bbox_path = kwargs['bboxs']
        landmark_path = kwargs['landmakes']
        if kwargs['split'] == 'train': # 训练集
            dataset = df.loc[df['Subject'] != kwargs['sub_id']]
        else: # 测试集
            dataset = df.loc[df['Subject'] == kwargs['sub_id']]


        Subject = dataset['Subject'].values
        Filename = dataset['Filename'].values
        OnsetFrame = dataset['OnsetFrame'].values

        ApexFrame = dataset['ApexFrame'].values
        OffsetFrame = dataset['OffsetFrame'].values
        Action_Units = dataset['Action Units'].values
        Estimated_Emotion = dataset['Estimated Emotion'].values

        # 得到数据和标签
        self.data_rgb, self.data_flow, self.label, self.domain_label, self.bboxs, self.landmarks = [], [], [], [], [], []
        self.onset, self.apexset, self.offset = [], [], []
        if self.data_name == 'mmew_wei':
            for i in range(len(Estimated_Emotion)):
                emotion = Estimated_Emotion[i]
                if emotion == 'happiness' or emotion == 'disgust' or emotion == 'fear' or emotion == 'sadness' or emotion == 'surprise':
                    self.data_rgb.append(os.path.join(root_rgb, Subject[i], Estimated_Emotion[i], Filename[i]))
                    self.data_flow.append(os.path.join(root_flow, Subject[i], Estimated_Emotion[i], Filename[i]))
                    self.label.append(emotion_to_label_5[emotion])
                    self.domain_label.append(int(1))
                    bboxs_txt_path = glob(os.path.join(bbox_path, Subject[i], Estimated_Emotion[i], Filename[i], '*'))
                    bbox = np.loadtxt(bboxs_txt_path[0])
                    self.bboxs.append(bbox)
                    landmarks_txt_path = glob(os.path.join(landmark_path, Subject[i], Estimated_Emotion[i], Filename[i], '*'))
                    landmark_5 = np.loadtxt(landmarks_txt_path[0])
                    self.landmarks.append(landmark_5)

                    self.onset.append(OnsetFrame[i])
                    self.apexset.append(ApexFrame[i])
                    self.offset.append(OffsetFrame[i])

        self.num_class_list = np.unique(self.label)

        self.transform = transform
        self.transform_norm = transform_norm

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img_on, img_apex, label, domain_label, landmark = self.get_data_from_index(idx)
        return img_on, img_apex, label, domain_label, landmark

    def get_data_from_index(self, index):

        path_onset = str(self.onset[index]) + '.jpg'
        path_apex = str(self.apexset[index]) + '.jpg'
        # /data/guoying/gyy/gyy_data/CK+/327labeled_CK+_cropped/happy/S037/
        '/data/guoying/gyy/gyy_data/CK+/327labeled_CK+_cropped/fear/S074/S074_001_00000020.png'
        path_onset = os.path.join(self.data_rgb[index], path_onset)
        path_apex = os.path.join(self.data_rgb[index], path_apex)

        path_flow = os.path.join(self.data_flow[index], 'img_00001.jpg')

        img_onset, img_apex = Image.open(path_onset).convert('RGB'), Image.open(path_apex).convert('RGB')

        img_flow = Image.open(path_flow).convert('RGB')

        label = self.label[index]
        domain_label = self.domain_label[index]
        bbox, landmark = self.bboxs[index], self.landmarks[index]

        ori_img_w, ori_img_h = img_onset.size

        # BoundingBox
        left = bbox[0]
        top = bbox[1]
        right = bbox[2]
        bottom = bbox[3]

        enlarge_bbox = True

        if self.data_name == 'mmew_wei':
            random_crop = True
        else:
            random_crop = False

        # Enlarge BoundingBox
        padding_w, padding_h = int(0.5 * max(0, int(0.20 * (right - left)))), int(
            0.5 * max(0, int(0.20 * (bottom - top))))

        if enlarge_bbox:
            left = max(left - padding_w, 0)
            right = min(right + padding_w, ori_img_w)

            top = max(top - padding_h, 0)
            bottom = min(bottom + padding_h, ori_img_h)

        if random_crop:
            x_offset = random.randint(-padding_w, padding_w)
            y_offset = random.randint(-padding_h, padding_h)

            left = max(left + x_offset, 0)
            right = min(right - x_offset, ori_img_w)

            top = max(top + y_offset, 0)
            bottom = min(bottom - y_offset, ori_img_h)

        img = img_onset.crop((left, top, right, bottom))
        crop_img_w, crop_img_h = img.size

        landmark[:, 0] -= left
        landmark[:, 1] -= top

        # Transform Image
        if self.transform is not None:
            img_onset = self.transform(img_onset)
            img_apex = self.transform(img_apex)
            img_flow = self.transform(img_flow)

        ALL = torch.cat((img_onset, img_apex), dim=0)
        ## data augmentation for training only
        if self.transform_norm is not None and self.split == 'train':
            ALL = self.transform_norm(ALL)
        img_onset = ALL[0:3, :, :]
        img_apex = ALL[3:6, :, :]

        inputSizeOfCropNet = 28
        landmark[:, 0] = landmark[:, 0] * inputSizeOfCropNet / crop_img_w
        landmark[:, 1] = landmark[:, 1] * inputSizeOfCropNet / crop_img_h
        landmark = landmark.astype(int)

        grid_len = 7
        half_grid_len = int(grid_len / 2)

        for index in range(landmark.shape[0]):
            if landmark[index, 0] <= (half_grid_len - 1):
                landmark[index, 0] = half_grid_len
            if landmark[index, 0] >= (inputSizeOfCropNet - half_grid_len):
                landmark[index, 0] = inputSizeOfCropNet - half_grid_len - 1
            if landmark[index, 1] <= (half_grid_len - 1):
                landmark[index, 1] = half_grid_len
            if landmark[index, 1] >= (inputSizeOfCropNet - half_grid_len):
                landmark[index, 1] = inputSizeOfCropNet - half_grid_len - 1

        return img_onset, img_apex, label, domain_label, landmark

class MMEW_wei_three(Dataset):

    def __init__(self, data_name, transform=None, transform_norm=None, **kwargs,):

        self.split = kwargs['split']
        self.data_name = data_name
        df = pd.read_excel(kwargs['excel'])
        # 将表格中的1-26修改为sub01-sub26，方便后面使用
        df['Subject'] = df['Subject'].apply(str)
        df['Subject'].replace({'1': 'S01', '2': 'S02', '3': 'S03', '4': 'S04', '5': 'S05',
                               '6': 'S06', '7': 'S07', '8': 'S08', '9': 'S09', '10': 'S10',
                               '11': 'S11', '12': 'S12', '13': 'S13', '14': 'S14', '15': 'S15',
                               '16': 'S16', '17': 'S17', '18': 'S18', '19': 'S19', '20': 'S20',
                               '21': 'S21', '22': 'S22', '23': 'S23', '24': 'S24', '25': 'S25',
                               '26': 'S26', '27': 'S27', '28': 'S28', '29': 'S29', '30': 'S30'
                               }, inplace=True)

        root_rgb = kwargs['path_rgb']
        root_flow = kwargs['path_flow']
        bbox_path = kwargs['bboxs']
        landmark_path = kwargs['landmakes']
        if kwargs['split'] == 'train': # 训练集
            dataset = df.loc[df['Subject'] != kwargs['sub_id']]
        else: # 测试集
            dataset = df.loc[df['Subject'] == kwargs['sub_id']]


        Subject = dataset['Subject'].values
        Filename = dataset['Filename'].values
        OnsetFrame = dataset['OnsetFrame'].values

        ApexFrame = dataset['ApexFrame'].values
        OffsetFrame = dataset['OffsetFrame'].values
        Action_Units = dataset['Action Units'].values
        Estimated_Emotion = dataset['Estimated Emotion'].values

        # 得到数据和标签
        self.data_rgb, self.data_flow, self.label, self.domain_label, self.bboxs, self.landmarks = [], [], [], [], [], []
        self.onset, self.apexset, self.offset = [], [], []
        if self.data_name == 'mmew_wei':
            for i in range(len(Estimated_Emotion)):
                emotion = Estimated_Emotion[i]
                if emotion == 'happiness' or emotion == 'disgust' or emotion == 'fear' or emotion == 'sadness' or emotion == 'surprise' or emotion == 'anger':
                    self.data_rgb.append(os.path.join(root_rgb, Subject[i], Estimated_Emotion[i], Filename[i]))
                    self.data_flow.append(os.path.join(root_flow, Subject[i], Estimated_Emotion[i], Filename[i]))
                    self.label.append(emotion_to_label3[emotion])
                    self.domain_label.append(int(1))
                    bboxs_txt_path = glob(os.path.join(bbox_path, Subject[i], Estimated_Emotion[i], Filename[i], '*'))
                    bbox = np.loadtxt(bboxs_txt_path[0])
                    self.bboxs.append(bbox)
                    landmarks_txt_path = glob(os.path.join(landmark_path, Subject[i], Estimated_Emotion[i], Filename[i], '*'))
                    landmark_5 = np.loadtxt(landmarks_txt_path[0])
                    self.landmarks.append(landmark_5)

                    self.onset.append(OnsetFrame[i])
                    self.apexset.append(ApexFrame[i])
                    self.offset.append(OffsetFrame[i])

        self.num_class_list = np.unique(self.label)

        self.transform = transform
        self.transform_norm = transform_norm

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img_on, img_apex, label, domain_label, landmark = self.get_data_from_index(idx)
        return img_on, img_apex, label, domain_label, landmark

    def get_data_from_index(self, index):

        path_onset = str(self.onset[index]) + '.jpg'
        path_apex = str(self.apexset[index]) + '.jpg'
        # /data/guoying/gyy/gyy_data/CK+/327labeled_CK+_cropped/happy/S037/
        '/data/guoying/gyy/gyy_data/CK+/327labeled_CK+_cropped/fear/S074/S074_001_00000020.png'
        path_onset = os.path.join(self.data_rgb[index], path_onset)
        path_apex = os.path.join(self.data_rgb[index], path_apex)

        path_flow = os.path.join(self.data_flow[index], 'img_00001.jpg')

        img_onset, img_apex = Image.open(path_onset).convert('RGB'), Image.open(path_apex).convert('RGB')

        img_flow = Image.open(path_flow).convert('RGB')

        label = self.label[index]
        domain_label = self.domain_label[index]
        bbox, landmark = self.bboxs[index], self.landmarks[index]

        ori_img_w, ori_img_h = img_onset.size

        # BoundingBox
        left = bbox[0]
        top = bbox[1]
        right = bbox[2]
        bottom = bbox[3]

        enlarge_bbox = True

        if self.data_name == 'mmew_wei':
            random_crop = True
        else:
            random_crop = False

        # Enlarge BoundingBox
        padding_w, padding_h = int(0.5 * max(0, int(0.20 * (right - left)))), int(
            0.5 * max(0, int(0.20 * (bottom - top))))

        if enlarge_bbox:
            left = max(left - padding_w, 0)
            right = min(right + padding_w, ori_img_w)

            top = max(top - padding_h, 0)
            bottom = min(bottom + padding_h, ori_img_h)

        if random_crop:
            x_offset = random.randint(-padding_w, padding_w)
            y_offset = random.randint(-padding_h, padding_h)

            left = max(left + x_offset, 0)
            right = min(right - x_offset, ori_img_w)

            top = max(top + y_offset, 0)
            bottom = min(bottom - y_offset, ori_img_h)

        img = img_onset.crop((left, top, right, bottom))
        crop_img_w, crop_img_h = img.size

        landmark[:, 0] -= left
        landmark[:, 1] -= top

        # Transform Image
        if self.transform is not None:
            img_onset = self.transform(img_onset)
            img_apex = self.transform(img_apex)
            img_flow = self.transform(img_flow)

        ALL = torch.cat((img_onset, img_apex), dim=0)
        ## data augmentation for training only
        if self.transform_norm is not None and self.split == 'train':
            ALL = self.transform_norm(ALL)
        img_onset = ALL[0:3, :, :]
        img_apex = ALL[3:6, :, :]

        inputSizeOfCropNet = 28
        landmark[:, 0] = landmark[:, 0] * inputSizeOfCropNet / crop_img_w
        landmark[:, 1] = landmark[:, 1] * inputSizeOfCropNet / crop_img_h
        landmark = landmark.astype(int)

        grid_len = 7
        half_grid_len = int(grid_len / 2)

        for index in range(landmark.shape[0]):
            if landmark[index, 0] <= (half_grid_len - 1):
                landmark[index, 0] = half_grid_len
            if landmark[index, 0] >= (inputSizeOfCropNet - half_grid_len):
                landmark[index, 0] = inputSizeOfCropNet - half_grid_len - 1
            if landmark[index, 1] <= (half_grid_len - 1):
                landmark[index, 1] = half_grid_len
            if landmark[index, 1] >= (inputSizeOfCropNet - half_grid_len):
                landmark[index, 1] = inputSizeOfCropNet - half_grid_len - 1

        return img_onset, img_apex, label, domain_label, landmark