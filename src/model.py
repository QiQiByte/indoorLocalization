import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
#哪里用了 KMeans
import numpy as np

class ECAAttention(nn.Module):
    def __init__(self, channels, k_size=3):
        super(ECAAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)


class CoordAttention(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x + shortcut


class CSILocalizationModel(nn.Module):
    def __init__(self, num_classes=317, train_coords=None, train_labels=None):
        super(CSILocalizationModel, self).__init__()

        # ⭐️ 不确定性加权参数
        self.log_var_cls = nn.Parameter(torch.zeros(1))
        self.log_var_reg = nn.Parameter(torch.zeros(1))

        if train_coords is not None and train_labels is not None:
            centers = self._init_centers(train_coords, train_labels, num_classes)
        else:
            centers = np.random.randn(num_classes, 2)
        self.register_buffer('class_centers', torch.tensor(centers, dtype=torch.float32))

        self.stem_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
        )

        self.block2 = nn.Sequential(
            ConvNeXtBlock(128),
            nn.Conv2d(128, 256, kernel_size=1),
            CoordAttention(256, 256)
        )

        self.block3 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvNeXtBlock(256),
            nn.Conv2d(256, 256, kernel_size=1),
            CoordAttention(256, 256)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.cls_proj = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )
        self.classifier = nn.Linear(256, num_classes)

        self.reg_proj = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.refiner = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        self.reg_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 64, kernel_size=1)
        )
        self.reg_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.reg_fc = nn.Linear(64 * 4 * 4, 2)

        self.coord_attn = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Softmax(dim=1)
        )

    def _init_centers(self, coords, labels, num_classes):
        kmeans = KMeans(n_clusters=num_classes, n_init=10)
        kmeans.fit(coords)
        return kmeans.cluster_centers_

    def forward(self, x):
        x = self.stem_block(x)
        x = self.block2(x)
        x = self.block3(x)

        global_feat = self.global_pool(x)
        global_feat = self.flatten(global_feat)

        cls_feat = self.cls_proj(global_feat)
        class_out = self.classifier(cls_feat)
        cls_probs = F.softmax(class_out, dim=1)
        coord_from_cls = torch.matmul(cls_probs, self.class_centers)

        reg_feat = self.reg_proj(global_feat)
        coord_offset = self.refiner(reg_feat)

        spatial_feat = self.reg_conv(x)
        spatial_feat = self.reg_pool(spatial_feat)
        spatial_feat = torch.flatten(spatial_feat, 1)
        coord_from_spatial = self.reg_fc(spatial_feat)

        coords_stack = torch.stack([coord_from_cls, coord_from_spatial, coord_offset], dim=1)
        fusion_input = coords_stack.view(coords_stack.size(0), -1)
        fusion_weights = self.coord_attn(fusion_input)
        coord_out = torch.sum(coords_stack * fusion_weights.unsqueeze(-1), dim=1)

        return class_out, coord_out

    def get_loss(self, class_out, coord_out, target_cls, target_coord):
        loss_cls = F.cross_entropy(class_out, target_cls)
        loss_reg = F.smooth_l1_loss(coord_out, target_coord)

        # ⭐️ 不确定性损失加权
        loss_cls_weighted = torch.exp(-self.log_var_cls) * loss_cls + self.log_var_cls
        loss_reg_weighted = torch.exp(-self.log_var_reg) * loss_reg + self.log_var_reg

        # 地理约束项（鼓励靠近类中心）
        with torch.no_grad():
            center_coords = self.class_centers[target_cls]
        loss_geo = 0.1 * torch.norm(coord_out - center_coords, dim=1).mean()

        total_loss = loss_cls_weighted + loss_reg_weighted + loss_geo

        return {
            'total_loss': total_loss,
            'cls_loss': loss_cls,
            'reg_loss': loss_reg,
            'geo_loss': loss_geo,
            'log_var_cls': self.log_var_cls.item(),
            'log_var_reg': self.log_var_reg.item()
        }

    def update_class_centers(self, pred_coords, target_cls, momentum=0.9):
        for i in range(pred_coords.size(0)):
            cls_idx = target_cls[i].item()
            self.class_centers[cls_idx] = (
                    momentum * self.class_centers[cls_idx] + (1 - momentum) * pred_coords[i]
            )

    def calculate_distance(self, pred_coords, true_coords):
        return torch.sqrt(torch.sum((pred_coords - true_coords) ** 2, dim=1)).detach()