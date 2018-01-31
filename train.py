#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch
import torchvision
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os

from SiameseNetwork.modelloader.siamese_net import SiameseNet
from SiameseNetwork.dataloader.orl_face_loader import OrlFaceLoader
from SiameseNetwork.loss import ContrastiveLoss



def train(args):
    transform = transforms.Compose([
                                     transforms.Scale((100,100)),
                                     transforms.ToTensor()])
    batch_size = 4
    dst =  OrlFaceLoader(root = os.path.expanduser('data/orl_faces/train'),
                                 transform = transform,
                                 should_invert = False)
    trainloader = DataLoader(dst, batch_size=batch_size, shuffle=True)

    model = SiameseNet()

    start_epoch = 0
    if args.resume_model_state_dict != '':
        start_epoch_id1 = args.resume_model_state_dict.rfind('_')
        start_epoch_id2 = args.resume_model_state_dict.rfind('.')
        start_epoch = int(args.resume_model_state_dict[start_epoch_id1 + 1:start_epoch_id2])
        model.load_state_dict(torch.load(args.resume_model_state_dict))

    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(),lr = 0.0005 )

    for epoch in range(start_epoch+1, 20000, 1):
        loss_epoch = 0
        data_count = 0
        for i, (img0s, img1s, labels) in enumerate(trainloader):
            # print(i)
            data_count = i

            img0s = Variable(img0s)
            img1s = Variable(img1s)
            labels = Variable(labels)

            outputs1, outputs2 = model(img0s, img1s)

            optimizer.zero_grad()
            loss = criterion(outputs1, outputs2, labels)
            loss_numpy = loss.data.numpy()
            loss_epoch += loss_numpy
            loss.backward()
            # print('loss:', loss_numpy)

            optimizer.step()
        loss_avg_epoch = loss_epoch / (data_count * 1.0)
        print(loss_avg_epoch)

        if args.save_model and epoch%args.save_epoch==0:
            torch.save(model.state_dict(), 'SiameseNet_orlface_{}.pt'.format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training parameter setting')
    parser.add_argument('--resume_model_state_dict', type=str, default='', help='resume model state dict path [ fcn32s_camvid_9.pt ]')
    parser.add_argument('--save_model', type=bool, default=False, help='save model [ False ]')
    parser.add_argument('--save_epoch', type=int, default=1, help='save model after epoch [ 1 ]')
    args = parser.parse_args()
    print(args)
    train(args)
