import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import *
from utils import *
import config
import gc


'''  
train
'''
device = torch.device('cuda:0')

#dataset
dirty_mnist_answer = pd.read_csv(config.train_data_answer_path)

# cross validation을 적용하기 위해 KFold 생성
kfold = KFold(n_splits=5, shuffle=True, random_state=0)

# dirty_mnist_answer에서 train_idx와 val_idx를 생성
for fold_index, (trn_idx, val_idx) in enumerate(kfold.split(dirty_mnist_answer), 1):
    print(f'[fold: {fold_index}]')
    # cuda cache clear

    gc.collect()
    torch.cuda.empty_cache()

    # train fold, validation fold 분할
    train_answer = dirty_mnist_answer.iloc[trn_idx]
    test_answer = dirty_mnist_answer.iloc[val_idx]

    # Dataset 정의
    train_dataset = DatasetMNIST(config.train_data_path, train_answer)
    valid_dataset = DatasetMNIST(config.train_data_path, test_answer)
    # DataLoader 정의
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=False,
        num_workers=0
    )
    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=config.valid_batch_size,
        shuffle=False,
        num_workers=0
    )

    # 모델 선언
    #model = MultiLabelResnet()
    model = Wide_resnet_Resnet()
    model.to(device)  # gpu에 모델 할당

    # 훈련 옵션 설정
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.75)
    criterion = torch.nn.BCELoss()

    # 훈련 시작
    valid_acc_max = 0
    for epoch in range(15):
        # 1개 epoch 훈련
        train_acc_list = []
        with tqdm(train_data_loader,  # train_data_loader를 iterative하게 반환
                  total=train_data_loader.__len__(),  # train_data_loader의 크기
                  unit="batch") as train_bar:  # 한번 반환하는 smaple의 단위는 "batch"

            for sample in train_bar:
                train_bar.set_description(f"Train Epoch {epoch}")
                # 갱신할 변수들에 대한 모든 변화도를 0으로 초기화
                optimizer.zero_grad()
                images, labels = sample['image'], sample['label']
                # tensor를 gpu에 올리기
                images = images.to(device)
                labels = labels.to(device)

                # 모델의 dropoupt, batchnormalization를 train 모드로 설정
                model.train()
                # .forward()에서 중간 노드의 gradient를 계산
                with torch.set_grad_enabled(True):
                    # 모델 예측
                    probs = model(images)
                    # loss 계산
                    loss = criterion(probs, labels)
                    # 중간 노드의 gradient로
                    # backpropagation을 적용하여
                    # gradient 계산
                    loss.backward()
                    # weight 갱신
                    optimizer.step()

                    # train accuracy 계산
                    probs = probs.cpu().detach().numpy()
                    labels = labels.cpu().detach().numpy()
                    preds = probs > 0.5
                    batch_acc = (labels == preds).mean()
                    train_acc_list.append(batch_acc)
                    train_acc = np.mean(train_acc_list)

                # 현재 progress bar에 현재 미니배치의 loss 결과 출력
                train_bar.set_postfix(train_loss=loss.item(),
                                      train_acc=train_acc)

        # 1개 epoch학습 후 Validation 점수 계산
        valid_acc_list = []
        with tqdm(valid_data_loader,
                  total=valid_data_loader.__len__(),
                  unit="batch") as valid_bar:
            for sample in valid_bar:
                valid_bar.set_description(f"Valid Epoch {epoch}")
                optimizer.zero_grad()
                images, labels = sample['image'], sample['label']
                images = images.to(device)
                labels = labels.to(device)

                # 모델의 dropoupt, batchnormalization를 eval모드로 설정
                model.eval()
                # .forward()에서 중간 노드의 gradient를 계산
                with torch.no_grad():
                    # validation loss만을 계산
                    probs = model(images)
                    valid_loss = criterion(probs, labels)

                    # train accuracy 계산
                    probs = probs.cpu().detach().numpy()
                    labels = labels.cpu().detach().numpy()
                    preds = probs > 0.5
                    batch_acc = (labels == preds).mean()
                    valid_acc_list.append(batch_acc)

                valid_acc = np.mean(valid_acc_list)
                valid_bar.set_postfix(valid_loss=valid_loss.item(),
                                      valid_acc=valid_acc)

        # Learning rate 조절
        lr_scheduler.step()

        # 모델 저장
        if valid_acc_max < valid_acc:
            valid_acc_max = valid_acc
            best_model = model
            # 모델을 저장할 구글 드라이브 경로
            torch.save(best_model, f'{config.model_path}{fold_index}_{config.model_name}_{valid_loss.item():2.4f}_epoch_{epoch}.pth')

    # 폴드별로 가장 좋은 모델 저장
    torch.save(best_model, f'{config.model_path}Best_{fold_index}_{config.model_name}_{valid_loss.item():2.4f}_epoch_{epoch}.pth')
    del model, best_model, images, labels, probs

'''
result analysis
'''
# gpu에 올라가 있는 tensor -> cpu로 이동 -> numpy array로 변환

def result_analysis(dirty_mnist_answer,
                    images,
                    sample_prob,
                    sample_labels,
                    idx) :
    sample_images = images.cpu().detach().numpy()
    plt.imshow(sample_images[idx][0])
    plt.title("sample input image")
    plt.show()
    print('예측값 : ',dirty_mnist_answer.columns[1:][sample_prob[idx] > 0.5])
    print('정답값 : ', dirty_mnist_answer.columns[1:][sample_labels[idx] > 0.5])
    plt.close()

'''
ensemble
'''
#test Dataset 정의
sample_submission = pd.read_csv(config.test_data_answer_path)
test_dataset = DatasetMNIST(config.test_data_path, sample_submission)


test_data_loader = DataLoader(
    test_dataset,
    batch_size = config.test_batch_size,
    shuffle = False,
    num_workers = 0,
    drop_last = False
)

predictions_list = []
# 5개의 fold마다 가장 좋은 모델을 이용하여 예측

# LOAD Best_models
model = Wide_resnet_Resnet()
model.to(device)  # gpu에 모델 할당


best_models = ['models/Best_3_wide_resnet101_2_0.6659_epoch_14.pth',
               'models/Best_1_wide_resnet101_2_0.6990_epoch_4.pth',
               'models/Best_1_wide_resnet101_2_0.6959_epoch_1.pth',
               'models/Best_2_wide_resnet101_2_0.7065_epoch_1.pth']

for model_ in best_models:
    # 0으로 채워진 array 생성
    prediction_array = np.zeros([sample_submission.shape[0],
                                 sample_submission.shape[1] - 1])
    for idx, sample in enumerate(test_data_loader):
        with torch.no_grad():
            # 추론
            model = torch.load(model_)
            model.eval()

            images = sample['image']
            images = images.to(device)
            probs = model(images)
            probs = probs.cpu().detach().numpy()
            preds = (probs > 0.5)

            # 예측 결과를
            # prediction_array에 입력
            batch_index = config.test_batch_size * idx
            prediction_array[batch_index: batch_index + images.shape[0], :] \
                = preds.astype(int)

    # 채널을 하나 추가하여 list에 append
    predictions_list.append(prediction_array[..., np.newaxis])

# axis = 2를 기준으로 평균
predictions_array = np.concatenate(predictions_list, axis = 2)
predictions_mean = predictions_array.mean(axis = 2)

# 평균 값이 0.5보다 클 경우 1 작으면 0
predictions_mean = (predictions_mean > 0.5) * 1
# 결과파일 만들기
sample_submission.iloc[:,1:] = predictions_mean
result_name = 'result/Wide_resnet_Resnet_prediction.csv'
sample_submission.to_csv(result_name, index = False)