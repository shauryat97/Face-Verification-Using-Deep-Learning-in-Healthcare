!pip install facenet-pytorch

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import pandas as pd


if torch.cuda.is_available():
    device = 'cuda:0'
else :
    device = 'cpu'

# Data Preprocessing - cropping out faces using MTCNN.
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval() # Pre-trained on vvgface2 for face embeddings. 



data_dir_post = '../input/dlcv-2021-hackathon/train/train/Post'
data_dir_pre = '../input/dlcv-2021-hackathon/train/train/Pre'
all_imgs_post = os.listdir(data_dir_post)
all_imgs_pre = os.listdir(data_dir_pre)
batch_size = 100

# Image DataLoader

def imageLoader(all_imgs_post,all_imgs_pre,batch_size,data_dir_post,data_dir_pre):
    for idx1 in range(0,len(all_imgs_post),batch_size):
        array_post = []
        array_pre = []
        for idx in range(idx1,idx1+batch_size):
            img_loc1 = os.path.join(data_dir_post,all_imgs_post[idx])
            img_loc2 = os.path.join(data_dir_pre,all_imgs_pre[idx])
            image1 = Image.open(img_loc1).convert("RGB")
            image2 = Image.open(img_loc2).convert("RGB")
            transform=transforms.Resize((512, 512))
            tensor1 = transform(image1)
            tensor2 = transform(image2)

            x_post_crop = mtcnn(tensor1)
            x_pre_crop = mtcnn(tensor2)
            if(np.shape(x_post_crop)):
                array_post.append(x_post_crop)
                array_pre.append(x_pre_crop)
        x_post = torch.stack(array_post)
        x_pre = torch.stack(array_pre)
        x_post_res = resnet(x_post)
        x_post_res = x_post_res.cpu().detach().numpy()
        x_pre_res = resnet(x_pre)
        x_pre_res = x_pre_res.cpu().detach().numpy()
        yield [x_post_res,x_pre_res]

# FC Layers.
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import binary_crossentropy


model = Sequential()
model.add(Dense(64, input_dim=1024, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training FC Layers.
epoch = 10
data = imageLoader(all_imgs_post,all_imgs_pre,batch_size,data_dir_post,data_dir_pre)
for ep in range(epoch):
    print("Data epoch",ep)
    x,y = next(data)
    x1 = x.copy()
    np.random.shuffle(x1)
    final_data = np.append(np.concatenate((x,x1)),np.concatenate((y,y)),axis=1)
    final_labels = np.zeros((len(x)+len(x1)))
    final_labels[0:len(x)]=np.ones((len(x)))
    model.fit(final_data,final_labels,epochs=20)

# Test DataLoader.
def imageLoader_test(all_imgs_post,all_imgs_pre,batch_size,data_dir_post,data_dir_pre):
    for idx1 in range(0,len(all_imgs_post),batch_size):
        array_post = []
        array_pre = []
        index_for_ex = []
        for idx in range(idx1,idx1+batch_size):
            print("tst",idx,end=" ")
            img_loc1 = os.path.join(data_dir_post,all_imgs_post[idx].strip())
            img_loc2 = os.path.join(data_dir_pre,all_imgs_pre[idx].strip())
            image1 = Image.open(img_loc1).convert("RGB")
            image2 = Image.open(img_loc2).convert("RGB")
            
            transform=transforms.Resize((512, 512))
            tensor1 = transform(image1)
            tensor2 = transform(image2)

            x_post_crop = mtcnn(tensor1)
            x_pre_crop = mtcnn(tensor2)
            if(np.shape(x_post_crop)):
                array_post.append(x_post_crop)
                array_pre.append(x_pre_crop)
                index_for_ex.append(idx)
        x_post = torch.stack(array_post)
        x_pre = torch.stack(array_pre)
        x_post_res = resnet(x_post)
        x_post_res = x_post_res.cpu().detach().numpy()
        x_pre_res = resnet(x_pre)
        x_pre_res = x_pre_res.cpu().detach().numpy()
        yield [x_post_res,x_pre_res,index_for_ex]

data_dir_test = '../input/dlcv-2021-hackathon/test'
test_csv = pd.read_csv("../input/dlcv-2021-hackathon/test.csv")
test_csv = test_csv.Id.str.split(",",expand=True) #0 is pre-op , 1 is post-op



batch_test_size = 10
test_data = imageLoader_test(list(test_csv[0]),list(test_csv[1]),batch_test_size, data_dir_test, data_dir_test)

# Inference
pred = []
pred_index = []
for i in range(0,5000,10):
    x,y,index_pred = next(test_data)
    final_test_data = np.append(x,y,axis=1)
    for te in model.predict(final_test_data):
        if te>=0.5:
            pred.append(1)
        else:
            pred.append(0)
    for te1 in index_pred:
        pred_index.append(te1)


# Creating the csv file for submission.
test_submission = pd.read_csv("../input/dlcv-2021-hackathon/sample_submission.csv")
predicted = [0]*5000
for i in range(len(pred_index)):
    if (pred[i]==1):
        predicted[pred_index[i]]=1
test_submission['Predicted']=predicted
test_submission.to_csv('team_submission.csv',index=False)

