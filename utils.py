import torch
import cv2
import numpy as np
import torchvision.transforms as T

class ToTensor(object):
    """numpy array를 tensor(torch)로 변환합니다."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.FloatTensor(image),
                'label': torch.FloatTensor(label)}


# numpy를 tensor로 변환하는 ToTensor 정의
def to_tensor() :
    return T.Compose([ToTensor()])

class DatasetMNIST(torch.utils.data.Dataset):
    def __init__(self,
                 dir_path,
                 meta_df,
                 transforms=to_tensor(),  # 미리 선언한 to_tensor를 transforms로 받음
                 augmentations=None):
        self.dir_path = dir_path  # 데이터의 이미지가 저장된 디렉터리 경로
        self.meta_df = meta_df  # 데이터의 인덱스와 정답지가 들어있는 DataFrame

        self.transforms = transforms  # Transform
        self.augmentations = augmentations  # Augmentation

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, index):
        # 폴더 경로 + 이미지 이름 + .png => 파일의 경로
        # 참고) "12".zfill(5) => 000012
        #       "146".zfill(5) => 000145
        # cv2.IMREAD_GRAYSCALE : png파일을 채널이 1개인 GRAYSCALE로 읽음
        image = cv2.imread(self.dir_path + \
                           str(self.meta_df.iloc[index, 0]).zfill(5) + '.png',
                           cv2.IMREAD_GRAYSCALE)
        # 0 ~ 255의 값을 갖고 크기가 (256,256)인 numpy array를
        # 0 ~ 1 사이의 실수를 갖고 크기가 (256,256,1)인 numpy array로 변환
        image = (image / 255).astype('float')[..., np.newaxis]

        # 정답 numpy array생성(존재하면 1 없으면 0)
        label = self.meta_df.iloc[index, 1:].values.astype('float')
        sample = {'image': image, 'label': label}

        # transform 적용
        # numpy to tensor
        if self.transforms:
            sample = self.transforms(sample)

        # sample 반환
        return sample

