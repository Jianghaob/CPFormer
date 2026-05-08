import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionModule(nn.Module):
    """
    跨注意力模块，包含三个阶段：
    1. Feature Grouping: 将输入特征分组
    2. Group Propagation: 组间交互传播
    3. Feature Ungrouping: 将分组特征还原
    """
    def __init__(self, embed_dim=64, num_heads=8, num_groups=4, dropout_attn=0., dropout_ffn=0.1):
        """
        Args:
            embed_dim: 嵌入维度 D
            num_heads: 多头注意力的头数 h
            num_groups: Group token的数量 M
            dropout: Dropout比率
        """
        super(CrossAttentionModule, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"
        
        # ========== Feature Grouping 阶段 ==========
        # 可学习的Group token G，shape=M D
        self.group_token = nn.Parameter(torch.randn(1, num_groups, embed_dim))
        
        # G经过线性层得到Q
        self.group_q_proj = nn.Linear(embed_dim, embed_dim)
        
        # 输入X经过线性层得到K和V
        self.group_k_proj = nn.Linear(embed_dim, embed_dim)
        self.group_v_proj = nn.Linear(embed_dim, embed_dim)
        
        # 多头注意力后的输出投影
        # self.group_out_proj = nn.Linear(embed_dim, embed_dim)
        self.group_dropout = nn.Dropout(dropout_attn)
        
        # ========== Group Propagation 阶段 ==========
        self.group_norm1 = nn.LayerNorm(embed_dim)
        # 组间交互的线性层（在D维度上操作）
        self.group_interaction = nn.Linear(num_groups, num_groups)
        self.group_norm2 = nn.LayerNorm(embed_dim)
        self.group_ffn = nn.Linear(embed_dim, embed_dim)
        self.group_dropout1 = nn.Dropout(dropout_ffn)
        self.group_dropout2 = nn.Dropout(dropout_ffn)
        
        # ========== Feature Ungrouping 阶段 ==========
        # 原始输入X经过线性层得到Q
        self.ungroup_q_proj = nn.Linear(embed_dim, embed_dim)
        # X_2经过线性层得到K、V
        self.ungroup_k_proj = nn.Linear(embed_dim, embed_dim)
        self.ungroup_v_proj = nn.Linear(embed_dim, embed_dim)
        
        # 多头注意力后的输出投影
        self.ungroup_out_proj = nn.Linear(embed_dim, embed_dim)
        self.ungroup_dropout = nn.Dropout(dropout_attn)
    
    def forward(self, x):
        """
        Args:
            x: 输入张量，形状为 B N D，其中N是序列维度，D是嵌入维度
        
        Returns:
            out: 输出张量，形状为 B N D
        """
        B, N, D = x.shape
        
        # ========== Feature Grouping 阶段 ==========
        # 引入可学习Group token G，扩张为B M D
        G = self.group_token.expand(B, -1, -1)  # B M D
        
        # G经过线性层得到Q
        Q_group = self.group_q_proj(G)  # B M D
        
        # 输入X经过线性层得到K和V
        K_group = self.group_k_proj(x)  # B N D
        V_group = self.group_v_proj(x)  # B N D
        
        # 使用多头注意力机制
        # Reshape为多头格式: B h M d_h 和 B h N d_h
        Q_group = Q_group.view(B, self.num_groups, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B h M d_h
        K_group = K_group.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B h N d_h
        V_group = V_group.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B h N d_h
        
        # 计算注意力分数
        scores = torch.matmul(Q_group, K_group.transpose(-2, -1)) / (self.head_dim ** 0.5)  # B h M N
        attn_weights = torch.softmax(scores, dim=-1)  # B h M N
        attn_weights = self.group_dropout(attn_weights)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, V_group)  # B h M d_h
        
        # 重塑为B M D
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # B M h d_h
        attn_output = attn_output.view(B, self.num_groups, D)  # B M D
        
        # 输出投影
        # grouped_features = self.group_out_proj(attn_output)  # B M D
        grouped_features = attn_output
        # ========== Group Propagation 阶段 ==========
        # 输入B M D，先经过LayerNorm
        x_norm = self.group_norm1(grouped_features)  # B M D
        
        # 转置得到B D M
        x_transposed = x_norm.transpose(1, 2)  # B D M
        
        # 线性层做组间交互
        x_interacted = self.group_interaction(x_transposed)  # B D M
        
        # 转置回B M D
        x_interacted = x_interacted.transpose(1, 2)  # B M D
        
        # 与输入残差连接得到X_1
        x_1 = grouped_features + self.group_dropout1(x_interacted)  # B M D
        
        # 再经过LayerNorm
        x_1_norm = self.group_norm2(x_1)  # B M D
        
        # 再经过线性层
        x_1_ffn = self.group_ffn(x_1_norm)  # B M D
        
        # 再与X_1残差连接，得到输出X_2
        x_2 = x_1 + self.group_dropout2(x_1_ffn)  # B M D
        
        # ========== Feature Ungrouping 阶段 ==========
        # 原始输入X经过线性层得到Q
        Q_ungroup = self.ungroup_q_proj(x)  # B N D
        
        # X_2经过线性层得到K、V
        K_ungroup = self.ungroup_k_proj(x_2)  # B M D
        V_ungroup = self.ungroup_v_proj(x_2)  # B M D
        
        # 使用多头注意力机制
        # Reshape为多头格式
        Q_ungroup = Q_ungroup.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B h N d_h
        K_ungroup = K_ungroup.view(B, self.num_groups, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B h M d_h
        V_ungroup = V_ungroup.view(B, self.num_groups, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B h M d_h
        
        # 计算注意力分数
        scores = torch.matmul(Q_ungroup, K_ungroup.transpose(-2, -1)) / (self.head_dim ** 0.5)  # B h N M
        attn_weights = torch.softmax(scores, dim=-1)  # B h N M
        attn_weights = self.ungroup_dropout(attn_weights)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, V_ungroup)  # B h N d_h
        
        # 重塑为B N D
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # B N h d_h
        attn_output = attn_output.view(B, N, D)  # B N D
        
        # 输出投影
        out = self.ungroup_out_proj(attn_output)  # B N D
        
        return out


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(self.avg_pool(x))
        return x * scale


class CAModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        proj = x.view(b, c, -1)  # B C (HW)
        energy = torch.matmul(proj, proj.transpose(1, 2))  # B C C
        attention = F.softmax(energy, dim=-1)
        out = torch.matmul(attention, proj).view(b, c, h, w)
        return self.gamma * out + x


class FeatureEnhancementBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.cam = CAModule(channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.cam(out)
        out = self.conv2(out)
        return out + residual


class SpectralSpatialExtractor(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.conv_expand = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True)
        )
        self.se = SEBlock(channels * 2, reduction)
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        out = self.conv_expand(x)
        out = self.se(out)
        out = self.conv_reduce(out)
        return out


class TransformerEncoderWithEnhance(nn.Module):
    def __init__(self, embed_dim=64, num_heads=8, num_groups=4,
                 dropout_attn=0., dropout_ffn=0.1):
        super().__init__()
        self.cross_att = CrossAttentionModule(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_groups=num_groups,
            dropout_attn=dropout_attn,
            dropout_ffn=dropout_ffn
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.feature_enhance = FeatureEnhancementBlock(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        b, c, h, w = x.shape
        seq = x.flatten(2).transpose(1, 2)  # B N C
        seq = seq + self.cross_att(seq)
        seq = self.norm1(seq)

        spatial = seq.transpose(1, 2).reshape(b, c, h, w)
        enhanced = self.feature_enhance(spatial)
        seq_enhanced = enhanced.flatten(2).transpose(1, 2)
        seq = seq + seq_enhanced
        seq = self.norm2(seq)
        return seq.transpose(1, 2).reshape(b, c, h, w)


class CWSSTNet(nn.Module):
    def __init__(self, in_channels, num_classes, embed_dim=64, num_heads=8,
                 num_groups=4, depth=2, dropout_attn=0., dropout_ffn=0.1, se_reduction=4):
        super().__init__()
        self.input_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1, bias=False)
        self.spec_spa = SpectralSpatialExtractor(embed_dim, reduction=se_reduction)
        self.encoders = nn.ModuleList([
            TransformerEncoderWithEnhance(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_groups=num_groups,
                dropout_attn=dropout_attn,
                dropout_ffn=dropout_ffn
            )
            for _ in range(depth)
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = x.permute(0,3,1,2) #->64 1 30 13 13
        x = self.input_proj(x)
        x = self.spec_spa(x)
        for encoder in self.encoders:
            x = encoder(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CWSSTNet(in_channels=16, num_classes=10).to(device)
    model.eval()
    dummy_input = torch.randn(4, 16, 13, 13, device=device)
    with torch.no_grad():
        output = model(dummy_input)
    print("Output shape:", output.shape)