# source: https://github.com/lironui/Double-Branch-Dual-Attention-Mechanism-Network/tree/master/DBDA
# Classification of hyperspectral image based on double-branch dual-attention mechanism network, Remote Sensing, 2020

import torch
from torch import nn
import math
import torch.nn.functional as F


class mish(nn.Module):
    def __init__(self):
        super(mish, self).__init__()

    # Also see https://arxiv.org/abs/1606.08415
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        # self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, depth, height, width = x.size()  # b 60 1 9 9
        proj_query = x.view(m_batchsize, C, -1)  # b 60 81
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)  # b 81 60
        energy = torch.bmm(proj_query, proj_key)  # b 60 60
        # torch.max： 返回最大值张量、最大值的索引张量，取第一个
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)  # b 60 60
        proj_value = x.view(m_batchsize, C, -1)  # b 60 81
        out = torch.bmm(attention, proj_value)  # b 60 81
        out = out.view(m_batchsize, C, depth, height, width)
        out = self.gamma * out + x  # C*H*W
        return out

class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x): # b 60 1 9 9
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, depth, height, width = x.size()  # b 60 1 9 9
        proj_query = self.query_conv(x).view(m_batchsize, -1, depth * height * width).permute(0, 2, 1)  # b 60 1 9 9 -- b 7 1 9 9 -- b 81 7
        proj_key = self.key_conv(x).view(m_batchsize, -1, depth * height * width)  # b 7 81
        energy = torch.bmm(proj_query, proj_key)  # b 81 81
        attention = self.softmax(energy)  # b 81 81
        proj_value = self.value_conv(x).view(m_batchsize, -1, depth * height * width)  # b 60 81
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # # b 60 81 b 81 81 -- b 60 81
        out = out.view(m_batchsize, C, depth, height, width)

        out = self.gamma * out + x
        return out

class DBDA_network_MISH(nn.Module):
    def __init__(self, band, classes):
        super(DBDA_network_MISH, self).__init__()

        # spectral branch
        self.name = 'DBDA_MISH'
        self.conv11 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, 7), stride=(1, 1, 2))
        # Dense block
        self.batch_norm11 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            mish()
        )
        self.conv12 = nn.Conv3d(in_channels=24, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm12 = nn.Sequential(
            nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
            mish()
        )
        self.conv13 = nn.Conv3d(in_channels=36, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm13 = nn.Sequential(
            nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
            mish()
        )
        self.conv14 = nn.Conv3d(in_channels=48, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm14 = nn.Sequential(
            nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),
            mish()
        )
        kernel_3d = math.floor((band - 6) / 2)
        self.conv15 = nn.Conv3d(in_channels=60, out_channels=60,
                                kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1))  # kernel size随数据变化

        # Spatial Branch
        self.conv21 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, band), stride=(1, 1, 1))
        # Dense block
        self.batch_norm21 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
            mish()
        )
        self.conv22 = nn.Conv3d(in_channels=24, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm22 = nn.Sequential(
            nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
            mish()
        )
        self.conv23 = nn.Conv3d(in_channels=36, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm23 = nn.Sequential(
            nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
            mish()
        )
        self.conv24 = nn.Conv3d(in_channels=48, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))

        self.batch_norm_spectral = nn.Sequential(
            nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),
            mish(),
            nn.Dropout(p=0.5)
        )
        self.batch_norm_spatial = nn.Sequential(
            nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),
            mish(),
            nn.Dropout(p=0.5)
        )

        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.full_connection = nn.Sequential(
            nn.Linear(120, classes)
        )

        self.attention_spectral = CAM_Module(60)
        self.attention_spatial = PAM_Module(60)

    def forward(self, X): # B 1 9 9 200

        # X = X.permute(0,1,4,2,3) # 16 1 200 9 9
        # spectral
        x11 = self.conv11(X) # 16 24 200 9 2?? # 24
        # print(x11.shape)
        # print('x11', x11.shape)
        x12 = self.batch_norm11(x11)
        x12 = self.conv12(x12) # 12
        # print('x12', x12.shape)

        x13 = torch.cat((x11, x12), dim=1) # 36
        # print('x13', x13.shape)
        x13 = self.batch_norm12(x13) #
        x13 = self.conv13(x13) # 12
        # print('x13', x13.shape)

        x14 = torch.cat((x11, x12, x13), dim=1) #
        x14 = self.batch_norm13(x14)
        x14 = self.conv14(x14)

        x15 = torch.cat((x11, x12, x13, x14), dim=1)
        # print('x15', x15.shape)

        x16 = self.batch_norm14(x15) # 16 60 200 9 2
        x16 = self.conv15(x16)
        # 光谱注意力通道
        x1 = self.attention_spectral(x16)
        x1 = x1 + x16

        # spatial
        # print('x', X.shape)
        x21 = self.conv21(X)
        x22 = self.batch_norm21(x21)
        x22 = self.conv22(x22)

        x23 = torch.cat((x21, x22), dim=1)
        x23 = self.batch_norm22(x23)
        x23 = self.conv23(x23)

        x24 = torch.cat((x21, x22, x23), dim=1)
        x24 = self.batch_norm23(x24)
        x24 = self.conv24(x24)

        x25 = torch.cat((x21, x22, x23, x24), dim=1) # b 60 1 9 9

        # 空间注意力机制
        x2 = self.attention_spatial(x25) # b 60 1 9 9
        x2 = x2 + x25

        # model1
        x1 = self.batch_norm_spectral(x1)
        x1 = self.global_pooling(x1)
        x1 = x1.squeeze(-1).squeeze(-1).squeeze(-1)
        x2 = self.batch_norm_spatial(x2)
        x2 = self.global_pooling(x2)
        x2 = x2.squeeze(-1).squeeze(-1).squeeze(-1)

        x_pre = torch.cat((x1, x2), dim=1)

        output = self.full_connection(x_pre)
        # output = self.fc(x_pre)
        return output


if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 模型参数配置
    band = 200  # 光谱波段数
    classes = 16  # 分类类别数
    batch_size = 4  # 批次大小
    patch_size = 9  # 空间补丁大小
    
    # 创建模型
    model = DBDA_network_MISH(band=band, classes=classes).to(device)
    model.eval()
    
    # 创建测试输入: (B, C, H, W, D) = (batch, 1, 9, 9, 200)
    test_input = torch.randn(batch_size, 1, patch_size, patch_size, band).to(device)
    print(f"\n输入形状: {test_input.shape}")
    
    # 前向传播测试
    print("\n开始前向传播...")
    try:
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✓ 前向传播成功！")
        print(f"输出形状: {output.shape}")
        print(f"期望输出形状: ({batch_size}, {classes})")
        
        # 验证输出形状
        assert output.shape == (batch_size, classes), \
            f"输出形状不匹配！期望: ({batch_size}, {classes}), 实际: {output.shape}"
        print("✓ 输出形状验证通过！")
        
        # 参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n模型参数统计:")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数量: {trainable_params:,}")
        
        # 测试反向传播
        print("\n测试反向传播...")
        model.train()
        test_input.requires_grad = True
        output = model(test_input)
        loss = output.sum()
        loss.backward()
        print("✓ 反向传播成功！")
        
        print("\n" + "="*50)
        print("所有测试通过！✓")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()