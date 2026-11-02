import os
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
import pandas as pd
from glob import glob
import numpy as np

emotion_to_label3 = {
                    "disgust": 0,
                    "fear": 0,
                    "sadness": 0,
                    "contempt": 0,
                    "anger": 0,
                    "happy": 1,
                    "surprise": 2,
                    }
# emotion_to_label3 = {
#                     "disgust": 0,
#                     "contempt": 0,
#                     "anger": 0,
#                     "happy": 1,
#                     "surprise": 2,
#                     }

emotion_to_label_5 = {"happy": 0,
                    "disgust": 1,
                    "fear": 2,
                    "sadness": 3,
                    "surprise": 4,
                    }

emotion_to_label6 = {"happy": 0,
                    "disgust": 1,
                    "fear": 2,
                    "sadness": 3,
                    "surprise": 4,
                    "anger": 5,
                    }

emotion_to_label7 = {"happy": 0,
                    "disgust": 1,
                    "fear": 2,
                    "sadness": 3,
                    "surprise": 4,
                    "anger": 5,
                    "contempt": 6
                    }

class CK_seven(Dataset):

    def __init__(self, data_name, transform=None, transform_norm=None, **kwargs,):

        self.split = kwargs['split']
        self.data_name = data_name
        df = pd.read_excel(kwargs['excel'])
        root_rgb = kwargs['path_rgb']
        root_flow = kwargs['path_flow']
        bbox_path = kwargs['bboxs']
        landmark_path = kwargs['landmakes']

        Subject = df['Subject'].values
        OnsetFrame = df['OnsetFrame'].values
        InterFrame = df['InterFrame'].values
        ApexFrame = df['ApexFrame'].values
        Estimated_Emotion = df['Estimated_Emotion'].values

        # 得到数据和标签
        self.data_rgb, self.data_flow, self.label, self.domain_label, self.bboxs, self.landmarks = [], [], [], [], [], []
        self.onset, self.interset, self.apexset = [], [], []
        if self.data_name == 'ck':
            for i in range(len(Estimated_Emotion)):
                emotion = Estimated_Emotion[i]
                if emotion == 'happy' or emotion == 'disgust' or emotion == 'fear' or emotion == 'sadness' or emotion == 'surprise' or emotion == 'anger' or emotion == 'contempt':
                    self.data_rgb.append(os.path.join(root_rgb, emotion, Subject[i]))
                    self.data_flow.append(os.path.join(root_flow, emotion, Subject[i]))
                    self.label.append(emotion_to_label7[emotion])
                    self.domain_label.append(int(0))
                    bboxs_txt_path = glob(os.path.join(bbox_path, emotion, Subject[i], '*'))
                    bbox = np.loadtxt(bboxs_txt_path[0])
                    self.bboxs.append(bbox)
                    landmarks_txt_path = glob(os.path.join(landmark_path, emotion, Subject[i], '*'))
                    landmark_5 = np.loadtxt(landmarks_txt_path[0])
                    self.landmarks.append(landmark_5)

                    self.onset.append(OnsetFrame[i])
                    self.interset.append(InterFrame[i])
                    self.apexset.append(ApexFrame[i])

        self.num_class_list = np.unique(self.label)

        self.transform = transform
        self.transform_norm = transform_norm

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img_on, img_apex, label, domain_label, landmark = self.get_data_from_index(idx)
        return img_on, img_apex, label, domain_label, landmark

    def get_data_from_index(self, index):

        path_onset = str(self.onset[index]) + '.png'
        path_apex = str(self.apexset[index]) + '.png'
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

        if self.data_name == 'ck':
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

class CK_six(Dataset):

    def __init__(self, data_name, transform=None, transform_norm=None, **kwargs,):

        self.split = kwargs['split']
        self.data_name = data_name
        df = pd.read_excel(kwargs['excel'])
        root_rgb = kwargs['path_rgb']
        root_flow = kwargs['path_flow']
        bbox_path = kwargs['bboxs']
        landmark_path = kwargs['landmakes']

        Subject = df['Subject'].values
        OnsetFrame = df['OnsetFrame'].values
        InterFrame = df['InterFrame'].values
        ApexFrame = df['ApexFrame'].values
        Estimated_Emotion = df['Estimated_Emotion'].values

        # 得到数据和标签
        self.data_rgb, self.data_flow, self.label, self.domain_label, self.bboxs, self.landmarks = [], [], [], [], [], []
        self.onset, self.interset, self.apexset = [], [], []
        if self.data_name == 'ck':
            for i in range(len(Estimated_Emotion)):
                emotion = Estimated_Emotion[i]
                if emotion == 'happy' or emotion == 'disgust' or emotion == 'fear' or emotion == 'sadness' or emotion == 'surprise' or emotion == 'anger':
                    self.data_rgb.append(os.path.join(root_rgb, emotion, Subject[i]))
                    self.data_flow.append(os.path.join(root_flow, emotion, Subject[i]))
                    self.label.append(emotion_to_label6[emotion])
                    self.domain_label.append(int(0))
                    bboxs_txt_path = glob(os.path.join(bbox_path, emotion, Subject[i], '*'))
                    bbox = np.loadtxt(bboxs_txt_path[0])
                    self.bboxs.append(bbox)
                    landmarks_txt_path = glob(os.path.join(landmark_path, emotion, Subject[i], '*'))
                    landmark_5 = np.loadtxt(landmarks_txt_path[0])
                    self.landmarks.append(landmark_5)

                    self.onset.append(OnsetFrame[i])
                    self.interset.append(InterFrame[i])
                    self.apexset.append(ApexFrame[i])

        self.num_class_list = np.unique(self.label)

        self.transform = transform
        self.transform_norm = transform_norm

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img_on, img_apex, label, domain_label, landmark = self.get_data_from_index(idx)
        return img_on, img_apex, label, domain_label, landmark

    def get_data_from_index(self, index):

        path_onset = str(self.onset[index]) + '.png'
        path_apex = str(self.apexset[index]) + '.png'
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

        if self.data_name == 'ck':
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

class CK_five(Dataset):

    def __init__(self, data_name, transform=None, transform_norm=None, **kwargs,):

        self.split = kwargs['split']
        self.data_name = data_name
        df = pd.read_excel(kwargs['excel'])
        root_rgb = kwargs['path_rgb']
        root_flow = kwargs['path_flow']
        bbox_path = kwargs['bboxs']
        landmark_path = kwargs['landmakes']

        Subject = df['Subject'].values
        OnsetFrame = df['OnsetFrame'].values
        InterFrame = df['InterFrame'].values
        ApexFrame = df['ApexFrame'].values
        Estimated_Emotion = df['Estimated_Emotion'].values

        # 得到数据和标签
        self.data_rgb, self.data_flow, self.label, self.domain_label, self.bboxs, self.landmarks = [], [], [], [], [], []
        self.onset, self.interset, self.apexset = [], [], []
        if self.data_name == 'ck':
            for i in range(len(Estimated_Emotion)):
                emotion = Estimated_Emotion[i]
                if emotion == 'happy' or emotion == 'disgust' or emotion == 'fear' or emotion == 'sadness' or emotion == 'surprise':
                    self.data_rgb.append(os.path.join(root_rgb, emotion, Subject[i]))
                    self.data_flow.append(os.path.join(root_flow, emotion, Subject[i]))
                    self.label.append(emotion_to_label_5[emotion])
                    self.domain_label.append(int(0))
                    bboxs_txt_path = glob(os.path.join(bbox_path, emotion, Subject[i], '*'))
                    bbox = np.loadtxt(bboxs_txt_path[0])
                    self.bboxs.append(bbox)
                    landmarks_txt_path = glob(os.path.join(landmark_path, emotion, Subject[i], '*'))
                    landmark_5 = np.loadtxt(landmarks_txt_path[0])
                    self.landmarks.append(landmark_5)

                    self.onset.append(OnsetFrame[i])
                    self.interset.append(InterFrame[i])
                    self.apexset.append(ApexFrame[i])

        self.num_class_list = np.unique(self.label)

        self.transform = transform
        self.transform_norm = transform_norm

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img_on, img_apex, label, domain_label, landmark = self.get_data_from_index(idx)
        return img_on, img_apex, label, domain_label, landmark

    def get_data_from_index(self, index):

        path_onset = str(self.onset[index]) + '.png'
        path_apex = str(self.apexset[index]) + '.png'
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

        if self.data_name == 'ck':
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

class CK_three(Dataset):

    def __init__(self, data_name, transform=None, transform_norm=None, **kwargs,):

        self.split = kwargs['split']
        self.data_name = data_name
        df = pd.read_excel(kwargs['excel'])
        root_rgb = kwargs['path_rgb']
        root_flow = kwargs['path_flow']
        bbox_path = kwargs['bboxs']
        landmark_path = kwargs['landmakes']

        Subject = df['Subject'].values
        OnsetFrame = df['OnsetFrame'].values
        InterFrame = df['InterFrame'].values
        ApexFrame = df['ApexFrame'].values
        Estimated_Emotion = df['Estimated_Emotion'].values

        # 得到数据和标签
        self.data_rgb, self.data_flow, self.label, self.domain_label, self.bboxs, self.landmarks = [], [], [], [], [], []
        self.onset, self.interset, self.apexset = [], [], []
        if self.data_name == 'ck':
            for i in range(len(Estimated_Emotion)):
                emotion = Estimated_Emotion[i]
                # if emotion == 'happy' or emotion == 'disgust' or emotion == 'surprise' or emotion == 'contempt' or emotion == 'anger':
                #
                if emotion == 'happy' or emotion == 'disgust' or emotion == 'fear' or emotion == 'sadness' or emotion == 'surprise' or emotion == 'contempt' or emotion == 'anger':
                    self.data_rgb.append(os.path.join(root_rgb, emotion, Subject[i]))
                    self.data_flow.append(os.path.join(root_flow, emotion, Subject[i]))
                    self.label.append(emotion_to_label3[emotion])
                    self.domain_label.append(int(0))
                    bboxs_txt_path = glob(os.path.join(bbox_path, emotion, Subject[i], '*'))
                    bbox = np.loadtxt(bboxs_txt_path[0])
                    self.bboxs.append(bbox)
                    landmarks_txt_path = glob(os.path.join(landmark_path, emotion, Subject[i], '*'))
                    landmark_5 = np.loadtxt(landmarks_txt_path[0])
                    self.landmarks.append(landmark_5)

                    self.onset.append(OnsetFrame[i])
                    self.interset.append(InterFrame[i])
                    self.apexset.append(ApexFrame[i])

        self.num_class_list = np.unique(self.label)

        self.transform = transform
        self.transform_norm = transform_norm

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img_on, img_apex, label, domain_label, landmark = self.get_data_from_index(idx)
        return img_on, img_apex, label, domain_label, landmark

    def get_data_from_index(self, index):

        path_onset = str(self.onset[index]) + '.png'
        path_apex = str(self.apexset[index]) + '.png'
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

        if self.data_name == 'ck':
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
