
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from mmpose.models.builder import build_loss
from mmpose.models.utils.ops import resize
from ..builder import HEADS
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=8, kernel_size=7, ca=True):
        super(CBAM, self).__init__()
        self.ca = ca
        self.channel_attention = ChannelAttention(in_channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        if self.ca:
            out = self.channel_attention(x) * x
        else:
            out = x
        out = self.spatial_attention(out) * out
        return out



@HEADS.register_module()
class WeightHeadBase(nn.Module):
    def __init__(self,
                 in_channels,
                 input_size=(64, 64),
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self.in_channels = in_channels
        self.loss = build_loss(loss_keypoint)

        self.intermediate = self.in_channels * input_size[0] * input_size[1]

        self.features = nn.Identity()
        self.predictor = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.intermediate, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1),
        )

    def save_gradient(self, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)



    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        x = self.features(x)
        if isinstance(x, list):
            x = x[0]
        x = x.view(x.size(0), self.intermediate)
        x = self.predictor(x)
        x = x.squeeze()
        return x


    def get_loss(self, outputs, targets, targets_weight):
        losses = dict()
        losses['mse_loss'] = self.loss(outputs.squeeze(), targets.squeeze())
        # losses['l1_loss'] = F.smooth_l1_loss(outputs.squeeze(), targets.squeeze())
        return losses


    def get_accuracy(self, outputs, targets, target_weight):
        targets.squeeze()
        outputs.squeeze()
        batch_size = outputs.numel() if outputs.numel() == 1 else outputs.size()[0]
        error = outputs - targets
        errorp = error / targets * 100
        return {
            "outputs": outputs.sum(),
            "targets": targets.sum(),
            "gap": error.abs().sum(),
            "AbsErrorP": errorp.abs().mean(),
            "ErrorP": errorp.mean(),
            "SingleMaxP": errorp.abs().max()
        }

    def init_weights(self):
        pass

    def inference_model(self, x, flip_pairs=None):
        output = self.forward(x)
        return output.detach().cpu().numpy()

    def decode(self, img_metas, outputs, img_size):
        target = np.array([img['target'] for img in img_metas])
        target = target.squeeze()
        error = (outputs - target)
        retdict = {'error':error,
                   'target':target,
                   'output':outputs,
                   'img_metas':img_metas}

        return retdict



@HEADS.register_module()
class WeightHead(WeightHeadBase):
    def __init__(self,
                 in_channels,
                 input_size=(64, 64),
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None):
        super(WeightHead, self).__init__(
            in_channels,
            input_size,
            loss_keypoint,
            train_cfg,
            test_cfg)

        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.predictor = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2304, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1),
        )


    def forward(self, x):
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.predictor(x)
        x = x.squeeze()
        return x



@HEADS.register_module()
class WeightHead3(WeightHeadBase):
    def __init__(self,
                 in_channels,
                 input_size=(64, 64),
                 last_channels=128,
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None):
        super(WeightHead3, self).__init__(
            in_channels,
            input_size,
            loss_keypoint,
            train_cfg,
            test_cfg)

        self.last_channels = last_channels
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, self.last_channels, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.predictor = nn.Linear(self.last_channels, 1)

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.predictor(x)
        x = x.squeeze()
        return x



@HEADS.register_module()
class WeightHead4(WeightHeadBase):
    def __init__(self,
                 in_channels,
                 input_size=(64, 64),
                 last_channels=128,
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None):
        super(WeightHead4, self).__init__(
            in_channels,
            input_size,
            loss_keypoint,
            train_cfg,
            test_cfg)

        self.last_channels = last_channels
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels,
                      self.last_channels,
                      kernel_size=17,
                      stride=7,
                      padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.last_channels, self.last_channels, kernel_size=7, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.predictor = nn.Linear(self.last_channels, 1)

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.predictor(x)
        x = x.squeeze()
        return x


@HEADS.register_module()
class WeightHead5(WeightHeadBase):
    """
    Product element wise, for a weight density musk.
    """
    def __init__(self,
                 in_channels,
                 input_size=(64, 64),
                 last_channels=128,
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None):
        super(WeightHead5, self).__init__(
            in_channels,
            input_size,
            loss_keypoint,
            train_cfg,
            test_cfg)

        self.last_channels = last_channels
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels,
                      self.last_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.predictor = nn.Linear(self.last_channels, 1)

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.predictor(x)
        x = x.squeeze()
        return x



@HEADS.register_module()
class WeightHead6(WeightHeadBase):
    """
    Use a sum operator rather Linear at prediction,
    A small kernel will overfit.
    """
    def __init__(self,
                 in_channels,
                 input_size=(64, 64),
                 last_channels=128,
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None):
        super(WeightHead6, self).__init__(
            in_channels,
            input_size,
            loss_keypoint,
            train_cfg,
            test_cfg)

        self.last_channels = last_channels
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels,
                      self.last_channels,
                      kernel_size=11,
                      stride=4,
                      padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = x.sum(dim=1)
        return x

@HEADS.register_module()
class WeightHead8(WeightHeadBase):
    """
    Use a sum operator rather Linear at prediction,
    A small kernel will overfit.
    """
    def __init__(self,
                 in_channels,
                 input_size=(64, 64),
                 last_channels=8,
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None):
        super(WeightHead8, self).__init__(
            in_channels,
            input_size,
            loss_keypoint,
            train_cfg,
            test_cfg)

        self.last_channels = last_channels
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels,
                      self.last_channels,
                      kernel_size=11,
                      stride=4,
                      padding=0,
                      groups=8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = x.sum(dim=1)
        return x


@HEADS.register_module()
class WeightHead10(WeightHeadBase):
    """
    Use a sum operator rather Linear at prediction,
    A small kernel will overfit.
    """
    def __init__(self,
                 in_channels,
                 input_size=(64, 64),
                 last_channels=8,
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None):
        super(WeightHead10, self).__init__(
            in_channels,
            input_size,
            loss_keypoint,
            train_cfg,
            test_cfg)

        self.last_channels = last_channels
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels,
                      self.last_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      groups=8),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(8),
        )

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = x.sum(dim=1)
        return x





@HEADS.register_module()
class WeightHead11(WeightHeadBase):
    """
    Based on WeightHead5, to add a gate
    """
    def __init__(self,
                 in_channels,
                 input_size=(64, 64),
                 last_channels=128,
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None):
        super(WeightHead11, self).__init__(
            in_channels,
            input_size,
            loss_keypoint,
            train_cfg,
            test_cfg)

        self.last_channels = last_channels
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels,
                      self.last_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            # nn.AdaptiveAvgPool2d(1),
        )


    def forward(self, x):
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = x.sum(dim=1)
        return x


@HEADS.register_module()
class WeightHead12(WeightHeadBase):
    """
    Base on weighthead5, just test the last_channels, 512 NOT WORK.
    """
    def __init__(self,
                 in_channels,
                 input_size=(64, 64),
                 last_channels=512,
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None):
        super(WeightHead12, self).__init__(
            in_channels,
            input_size,
            loss_keypoint,
            train_cfg,
            test_cfg)

        self.last_channels = last_channels
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels,
                      self.last_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.ReLU6(inplace=True),
            nn.Dropout(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.predictor = nn.Linear(self.last_channels, 1)

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.predictor(x)
        x = x.squeeze()
        return x


@HEADS.register_module()
class WeightHead13(WeightHeadBase):
    """
    Base on weighthead5, just test the double Linear
    """
    def __init__(self,
                 in_channels,
                 input_size=(64, 64),
                 last_channels=128,
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None):
        super(WeightHead13, self).__init__(
            in_channels,
            input_size,
            loss_keypoint,
            train_cfg,
            test_cfg)

        self.last_channels = last_channels
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels,
                      self.last_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.ReLU6(inplace=True),
            nn.Dropout(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.predictor = nn.Sequential(
            nn.Linear(self.last_channels, self.last_channels),
            nn.ReLU6(inplace=True),
            nn.Linear(self.last_channels, 1),
        )

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.predictor(x)
        x = x.squeeze()
        return x


@HEADS.register_module()
class WeightHead14(WeightHeadBase):
    """
    Base on weighthead5, just test the double Linear
    """
    def __init__(self,
                 in_channels,
                 input_size=(64, 64),
                 last_channels=128,
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None):
        super(WeightHead14, self).__init__(
            in_channels,
            input_size,
            loss_keypoint,
            train_cfg,
            test_cfg)

        self.last_channels = last_channels
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels,
                      self.last_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.ReLU6(inplace=True),
            nn.Dropout(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.predictor = nn.Sequential(
            nn.Linear(self.last_channels, 4 * self.last_channels),
            nn.ReLU(inplace=True),
            nn.Linear(4 * self.last_channels, self.last_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.last_channels, 1),
        )

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.predictor(x)
        x = x.squeeze()
        return x



@HEADS.register_module()
class WeightHead15(WeightHeadBase):
    """
    Product element wise, for a weight density musk.
    """
    def __init__(self,
                 in_channels,
                 input_size=(64, 64),
                 last_channels=1024,
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None):
        super(WeightHead15, self).__init__(
            in_channels,
            input_size,
            loss_keypoint,
            train_cfg,
            test_cfg)

        self.last_channels = last_channels
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels,
                      self.last_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.predictor = nn.Linear(self.last_channels, 1)

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            x = x[0]
        ## RESHAPE
        N, C, H, W = x.size()
        x = x.reshape(N, -1, 16, 16)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.predictor(x)
        x = x.squeeze()
        return x


@HEADS.register_module()
class WeightHead16(WeightHeadBase):
    """
    Product element wise, for a weight density musk.
    """
    def __init__(self,
                 in_channels,
                 input_size=(64, 64),
                 last_channels=1024,
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None):
        super(WeightHead16, self).__init__(
            in_channels,
            input_size,
            loss_keypoint,
            train_cfg,
            test_cfg)

        self.last_channels = last_channels
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels,
                      self.last_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.predictor = nn.Linear(self.last_channels, 1)

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            x = x[0]
        ## RESHAPE
        N, C, H, W = x.size()
        x = x.reshape(N, -1, 8, 8)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.predictor(x)
        x = x.squeeze()
        return x


@HEADS.register_module()
class WeightHead17(WeightHeadBase):
    """
    WeighHead5 but deep feature.
    """
    def __init__(self,
                 in_channels,
                 input_size=(64, 64),
                 last_channels=128,
                 feature_deep=5,
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None):
        super(WeightHead17, self).__init__(
            in_channels,
            input_size,
            loss_keypoint,
            train_cfg,
            test_cfg)

        self.last_channels = last_channels
        self.feature_deep = feature_deep
        self.prefeatures = nn.Conv2d(self.in_channels,
                                     self.last_channels,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        self.features = nn.Sequential(
            nn.Conv2d(self.last_channels,
                      self.last_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.ReLU(inplace=True),
        )

        self.postfeatures = nn.Sequential(
            nn.Dropout(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.predictor = nn.Linear(self.last_channels, 1)

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = self.prefeatures(x)
        for _ in range(self.feature_deep):
            x = self.features(x)
        x = self.postfeatures(x)
        x = x.view(x.size(0), -1)
        x = self.predictor(x)
        x = x.squeeze()
        return x


@HEADS.register_module()
class WeightDepthHead1(WeightHeadBase):
    """
    Use a sum operator rather Linear at prediction,
    A small kernel will overfit.
    """
    def __init__(self,
                 in_channels,
                 input_size=(64, 64),
                 last_channels=128,
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None):
        super(WeightDepthHead1, self).__init__(
            in_channels,
            input_size,
            loss_keypoint,
            train_cfg,
            test_cfg)

        self.last_channels = last_channels
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels,
                      self.last_channels,
                      kernel_size=11,
                      stride=4,
                      padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        x, depth = x[0], x[1]
        # print("-----------------", type(x), len(x) if isinstance(x, list) else x.shape, depth.shape)
        if isinstance(x, (tuple, list)):
            x = x[0]
        depth = resize(depth, x[0].shape[1:])
        x = x.mul(depth)
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = x.sum(dim=1)
        return x


@HEADS.register_module()
class WeightDepthHead2(WeightHeadBase):
    """
    Use a sum operator rather Linear at prediction,
    A small kernel will overfit.
    """
    def __init__(self,
                 in_channels,
                 input_size=(64, 64),
                 last_channels=128,
                 num_stages=3,
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None):
        super(WeightDepthHead2, self).__init__(
            in_channels,
            input_size,
            loss_keypoint,
            train_cfg,
            test_cfg)

        self.num_stages = num_stages
        self.last_channels = last_channels
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels,
                      self.in_channels, #self.last_channels,
                      kernel_size=13,
                      stride=1,
                      padding=6),
            nn.ReLU(inplace=True)
        )
        self.predictor = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x, depth = x[0], x[1]
        # Get original x out
        if isinstance(x, (tuple, list)):
            x = x[0]
        # Get x out ([resnet, regnet, resnest] output a tuple)
        if isinstance(x, (tuple, list)):
            x = x[0]
        # print("-----------------", type(x), len(x) if isinstance(x, list) else x.shape, depth.shape)
        for _ in range(self.num_stages):
            depth = resize(depth, x[0].shape[1:])
            x = x.mul(depth)
            x = self.features(x)
        x = self.predictor(x)
        x = x.view(x.size(0), -1)
        x = x.sum(dim=1)
        return x


@HEADS.register_module()
class WeightDepthHead4(WeightHeadBase):
    """
    Use a sum operator rather Linear at prediction,
    A small kernel will overfit.
    from mmpose.models.heads import WeightDepthHead4
    model  = WeightDepthHead4(
        40,
        loss_keypoint = dict(type="MSELoss")
    )
    """
    def __init__(self,
                 in_channels,
                 input_size=(64, 64),
                 last_channels=128,
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None):
        super(WeightDepthHead4, self).__init__(
            in_channels,
            input_size,
            loss_keypoint,
            train_cfg,
            test_cfg)

        self.last_channels = last_channels
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels,
                      self.last_channels,
                      kernel_size=11,
                      stride=1,
                      padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.depthfeature = nn.Sequential(
            nn.Conv2d(1,
                      1,
                      kernel_size=11,
                      stride=1,
                      padding=5),
            nn.Conv2d(1,
                      1,
                      kernel_size=3,
                      stride=1,
                      padding=1),
        )



    def forward(self, x):
        x, depth = x[0], x[1]
        # print("-----------------", type(x), len(x) if isinstance(x, list) else x.shape, depth.shape)
        if isinstance(x, (tuple, list)):
            x = x[0]
        depth = resize(depth, x[0].shape[1:])
        depth = self.depthfeature(depth)
        x = x.mul(depth)
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = x.sum(dim=1)
        return x




@HEADS.register_module()
class WeightDepthHead5(WeightHeadBase):
    """
    Use a sum operator rather Linear at prediction,
    A small kernel will overfit.
    from mmpose.models.heads import WeightDepthHead5
    model  = WeightDepthHead5(
        40,
        loss_keypoint = dict(type="MSELoss")
    )
    """
    def __init__(self,
                 in_channels,
                 input_size=(64, 64),
                 last_channels=128,
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None):
        super(WeightDepthHead5, self).__init__(
            in_channels,
            input_size,
            loss_keypoint,
            train_cfg,
            test_cfg)

        self.last_channels = last_channels
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels,
                      self.last_channels,
                      kernel_size=11,
                      stride=1,
                      padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.depthfeature1 = nn.Conv2d(1,
                                       1,
                                       kernel_size=11,
                                       stride=1,
                                       padding=5)
        self.depthfeature2 = nn.Conv2d(1,
                                       1,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        self.cbamx = CBAM(self.in_channels, ratio=8, kernel_size=7, ca=False)
        self.cbamdepth = CBAM(1, ratio=1, kernel_size=7)

    def fuse(self, x, depth):

        depth = resize(depth, x[0].shape[1:])
        depthout1 = self.depthfeature1(depth)
        depthout2 = self.depthfeature2(depth)
        depth = depthout1 + depthout2

        outx = self.cbamx(x)
        outdepth = self.cbamdepth(depth)

        out = outx.mul(outdepth)

        return out

    def predictor(self, x):
        x = x.view(x.size(0), -1)
        x = x.sum(dim=1)
        return x

    def forward(self, x):
        x, depth = x[0], x[1]
        # print("-----------------", type(x), len(x) if isinstance(x, list) else x.shape, depth.shape)
        if isinstance(x, (tuple, list)):
            x = x[0]
        if isinstance(x, (tuple, list)):
            x = x[0]

        x = self.fuse(x, depth)
        x = self.features(x)
        x = self.predictor(x)
        return x
