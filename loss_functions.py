from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F


class Vgg19_pc(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19_pc, self).__init__()
        vgg_pretrained_features = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        vgg_pretrained_features = nn.DataParallel(vgg_pretrained_features.cuda())

        # this has Vgg config E:
        # partial convolution paper uses up to pool3
        # [64,'r', 64,r, 'M', 128,'r', 128,r, 'M', 256,'r', 256,r, 256,r, 256,r, 'M', 512,'r', 512,r, 512,r, 512,r]
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        n_new = 0
        for x in range(5):  # pool1,
            self.slice1.add_module(str(n_new), vgg_pretrained_features.module[x])
            n_new += 1
        for x in range(5, 10):  # pool2
            self.slice2.add_module(str(n_new), vgg_pretrained_features.module[x])
            n_new += 1
        for x in range(10, 19):  # pool3
            self.slice3.add_module(str(n_new), vgg_pretrained_features.module[x])
            n_new += 1
        for x in range(19, 28):  # pool4
            self.slice4.add_module(str(n_new), vgg_pretrained_features.module[x])
            n_new += 1
        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, x, full=False):
        h_relu1_2 = self.slice1(x)
        h_relu2_2 = self.slice2(h_relu1_2)
        h_relu3_4 = self.slice3(h_relu2_2)
        if full:
            h_relu4_4 = self.slice4(h_relu3_4)
            return h_relu1_2, h_relu2_2, h_relu3_4, h_relu4_4
        else:
            return h_relu1_2, h_relu2_2, h_relu3_4


vgg = Vgg19_pc()


def perceptual_loss(out_vgg, label_vgg, layer=None):
    if layer is not None:
        l_p = torch.mean((out_vgg[layer] - label_vgg[layer]) ** 2)
    else:
        l_p = 0
        for i in range(3):
            l_p += torch.mean((out_vgg[i] - label_vgg[i]) ** 2)

    return l_p


def smoothness(img, disp, gamma=1, mask=1):
    B, C, H, W = img.shape

    gray_img = getGrayscale(img)

    # Disparity smoothness
    sx_filter = torch.autograd.Variable(torch.Tensor([[0, 0, 0], [-1, 2, -1], [0, 0, 0]])).unsqueeze(
        0).unsqueeze(0).cuda()
    sy_filter = torch.autograd.Variable(torch.Tensor([[0, -1, 0], [0, 2, 0], [0, -1, 0]])).unsqueeze(
        0).unsqueeze(0).cuda()
    dx_filter = torch.autograd.Variable(torch.Tensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]])).unsqueeze(
        0).unsqueeze(0).cuda()
    dy_filter = torch.autograd.Variable(torch.Tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]])).unsqueeze(
        0).unsqueeze(0).cuda()
    dx1_filter = torch.autograd.Variable(torch.Tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])).unsqueeze(
        0).unsqueeze(0).cuda()
    dy1_filter = torch.autograd.Variable(torch.Tensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]])).unsqueeze(
        0).unsqueeze(0).cuda()
    dx_img = F.conv2d(gray_img, sx_filter, padding=1, stride=1)
    dy_img = F.conv2d(gray_img, sy_filter, padding=1, stride=1)
    dx_d = F.conv2d(disp, dx_filter, padding=1, stride=1)
    dy_d = F.conv2d(disp, dy_filter, padding=1, stride=1)
    dx1_d = F.conv2d(disp, dx1_filter, padding=1, stride=1)
    dy1_d = F.conv2d(disp, dy1_filter, padding=1, stride=1)
    Cds = torch.mean(
        mask * (torch.abs(dx_d) + torch.abs(dx1_d)) * torch.exp(-gamma * torch.abs(dx_img)) +
        mask * (torch.abs(dy_d) + torch.abs(dy1_d)) * torch.exp(-gamma * torch.abs(dy_img)))
    return Cds


def getGrayscale(input):
    # Input is mini-batch N x 3 x H x W of an RGB image (analog rgb from 0...1)
    output = torch.autograd.Variable(input.data.new(*input.size()))
    # Output is mini-batch N x 3 x H x W from y = 0 ... 1
    output[:, 0, :, :] = 0.299 * input[:, 0, :, :] + 0.587 * input[:, 1, :, :] + 0.114 * input[:, 2, :, :]
    return output[:, 0, :, :].unsqueeze(1)