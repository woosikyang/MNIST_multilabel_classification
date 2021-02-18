import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch

# nn.Module을 상속 받아 MultiLabelResnet를 정의
class MultiLabelResnet(nn.Module):
    def __init__(self):
        super(MultiLabelResnet, self).__init__()
        self.conv2d = nn.Conv2d(1, 3, 3, stride=1)
        self.resnet = models.resnet18()
        self.FC = nn.Linear(1000, 26)

    def forward(self, x):
        # resnet의 입력은 [3, N, N]으로
        # 3개의 채널을 갖기 때문에
        # resnet 입력 전에 conv2d를 한 층 추가
        x = F.relu(self.conv2d(x))

        # resnet18을 추가
        x = F.relu(self.resnet(x))

        # 마지막 출력에 nn.Linear를 추가
        # multilabel을 예측해야 하기 때문에
        # softmax가 아닌 sigmoid를 적용
        x = torch.sigmoid(self.FC(x))
        return x