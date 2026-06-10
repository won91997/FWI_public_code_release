from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F

l1loss = nn.L1Loss()
l2loss = nn.MSELoss()


def calculate_edge(edges):
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float).to('cuda')
    loss_out_w = edges.to(torch.float)
    dilated_tensor = F.conv2d(loss_out_w, kernel, padding=1, stride=1)
    result = torch.zeros_like(dilated_tensor)
    result[dilated_tensor != 0] = 1
    return result


def criterion_inv(outputs, vmodels):
    loss_g2v = l2loss(outputs, vmodels)
    return loss_g2v


def criterion_fcn(outputs, vmodels):
    loss_g1v = l1loss(outputs, vmodels)
    return loss_g1v


def criterion_12(outputs, vmodels):
    loss_glv = l1loss(outputs, vmodels)
    loss_g2v = l2loss(outputs, vmodels)
    result_loss = loss_glv + loss_g2v
    return result_loss


def criterion_tu(outputs, vmodels, edges):
    loss_glv = l1loss(outputs, vmodels)
    loss_g2v = l2loss(outputs, vmodels)
    loss_pixel = 0.5 * loss_glv + 0.5 * loss_g2v
    edge_weight = calculate_edge(edges)
    loss_edge = torch.sum(edge_weight * torch.abs(outputs-vmodels)) / (vmodels.size(0) * torch.sum(edge_weight))
    result_loss = 0.7 * loss_pixel + 0.3 * loss_edge
    return result_loss


class LossDDNet:
    def __init__(self, weights=[1, 1], entropy_weight=[1, 1]):
        '''
        Define the loss function of DDNet

        :param weights:         The weights of the two decoders in the calculation of the loss value.
        :param entropy_weight:  The weights of the two output channels in the second decoder.
        '''

        self.criterion1 = nn.MSELoss()
        ew = torch.from_numpy(np.array(entropy_weight).astype(np.float32)).cuda()
        self.criterion2 = nn.CrossEntropyLoss(weight=ew)    # For multi-classification, the current issue is a binary problem (either black or white).
        self.weights = weights

    def __call__(self, outputs1, outputs2, targets1, targets2):
        '''

        :param outputs1: Output of the first decoder
        :param outputs2: Velocity model
        :param targets1: Output of the second decoder
        :param targets2: Profile of the speed model
        :return:
        '''
        mse = self.criterion1(outputs1, targets1)
        cross = self.criterion2(outputs2, torch.squeeze(targets2).long())

        criterion = (self.weights[0] * mse + self.weights[1] * cross)

        return criterion


class Wasserstein_GP(nn.Module):
    def __init__(self, device, lambda_gp):
        super(Wasserstein_GP, self).__init__()
        self.device = device
        self.lambda_gp = lambda_gp

    def forward(self, real, fake, model):
        gradient_penalty = self.compute_gradient_penalty(model, real, fake)
        loss_real = torch.mean(model(real))
        loss_fake = torch.mean(model(fake))
        loss = -loss_real + loss_fake + gradient_penalty * self.lambda_gp
        return loss, loss_real-loss_fake, gradient_penalty

    def compute_gradient_penalty(self, model, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = model(interpolates)
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(real_samples.size(0), d_interpolates.size(1)).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


def criterion_g(pred, gt, net_d=None):
    l1loss = nn.L1Loss()
    l2loss = nn.MSELoss()
    loss_g1v = l1loss(pred, gt)
    loss_g2v = l2loss(pred, gt)
    loss = 100 * loss_g1v + 100 * loss_g2v
    if net_d is not None:
        loss_adv = -torch.mean(net_d(pred))
        loss += loss_adv
    return loss, loss_g1v, loss_g2v
