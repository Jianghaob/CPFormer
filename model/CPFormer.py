import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange
from collections import OrderedDict
from torch.nn.init import trunc_normal_

def _weights_init(m):
    """权重初始化函数"""
    if isinstance(m, (nn.Linear, nn.Conv3d)):
        init.kaiming_normal_(m.weight)


class Residual(nn.Module):
    """残差连接模块"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class LayerNormalize(nn.Module):
    """层归一化包装器"""
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MLP_Block(nn.Module):
    """前馈神经网络模块"""
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)



class DecayPos1d(nn.Module): # DecayPos1d(dim, heads, 2, 4)
    """一维衰减位置编码"""
    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        super().__init__()
        self.num_heads = num_heads
        
        # 预计算角度和衰减值
        # angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2)) # 每个头分到部分维度，取一半用于频率
        # angle = angle.unsqueeze(-1).repeat(1, 2).flatten() # [embed_dim // num_heads]
        decay = torch.log(1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        
        # self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)
        
    def generate_1d_decay(self, l: int):
        """生成1D衰减mask"""
        index = torch.arange(l, device=self.decay.device) # 0 1 2 3 4
        mask = (index[:, None] - index[None, :]).abs()  # [l, l] 
        return mask * self.decay[:, None, None]  # [n, l, l] 
    
    def forward(self, slen):
        return self.generate_1d_decay(slen)

class DecayPos1d_2(nn.Module): # DecayPos1d(dim, heads, 2, 4)
    """一维衰减位置编码"""
    def __init__(self, embed_dim, num_heads, initial_value, heads_range,init_tau=1.0):
        super().__init__()
        self.num_heads = num_heads
        self.init_tau = nn.Parameter(torch.tensor(3.0))
        # 预计算角度和衰减值
        # angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2)) # 每个头分到部分维度，取一半用于频率
        # angle = angle.unsqueeze(-1).repeat(1, 2).flatten() # [embed_dim // num_heads]
        decay = torch.log(1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        
        # self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)
        
    def generate_1d_decay(self, l: int):
        """生成1D衰减mask"""
        index = torch.arange(l, device=self.decay.device) # 0 1 2 3 4
        mask = (index[:, None] - index[None, :]).abs()  # [l, l] 
        mask = (mask / self.init_tau).exp()
        return mask * self.decay[:, None, None]  # [n, l, l] 负数 * 距离绝对值
    
    def forward(self, slen):
        return self.generate_1d_decay(slen)

def gaussian_spatial_position(h, w, sigma=1.0, device=None):
    """
    生成高斯空间位置权重矩阵
    
    Args:
        h: 高度
        w: 宽度
        sigma: 高斯分布标准差
        device: 计算设备
    
    Returns:
        形状为[h, w]的高斯权重矩阵
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 计算中心坐标
    center_h, center_w = h // 2, w // 2
    
    # 创建坐标网格并计算到中心的距离
    y_coords = torch.arange(h, dtype=torch.float32, device=device).view(-1, 1)
    x_coords = torch.arange(w, dtype=torch.float32, device=device).view(1, -1)
    
    # 计算高斯权重
    dist_sq = (y_coords - center_h) ** 2 + (x_coords - center_w) ** 2
    return torch.exp(-dist_sq / (2 * sigma ** 2))


class CenterPixelAttention(nn.Module):
    """
    基于中心像素的注意力机制，包含空间注意力和光谱注意力
    """
    def __init__(self, dim, heads, dim_heads, dropout):
        super().__init__()
        inner_dim = dim_heads * heads
        self.heads = heads
        self.dim = dim
        self.inner_dim = inner_dim
        self.dim_heads = dim_heads
        self.scale = dim_heads**(-0.5)
        
        # 中心像素注意力的QKV映射
        self.to_qkv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_q_center = nn.Linear(dim, inner_dim, bias=False)
        
        # 光谱注意力的映射
        self.q_lr_self = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.Conv2d(dim, inner_dim, kernel_size=1)
        )
        self.self_attn_k = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.Conv2d(dim, inner_dim, kernel_size=1)
        )
        self.self_attn_v = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.Conv2d(dim, inner_dim, kernel_size=1)
        )
        
        # 位置编码和权重参数
        self.realPos = DecayPos1d_2(dim, heads, 2, 4)
        # self.realPos = LearnableDecayPos1d(heads)
        self.alpha = nn.Parameter(torch.ones(1)*0.01)
        # self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        # 输出层
        self.final_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        b, n, d = x.shape
        h = self.heads
        h1 = w1 = int(math.sqrt(n))
        center_idx = n // 2
        
        # 预计算高斯位置权重
        g_pos = gaussian_spatial_position(h1, w1, device=x.device)
        g_pos = g_pos.view(-1).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        # 第一阶段：中心像素空间注意力
        center_pixel = x[:, center_idx:center_idx+1, :]
        
        # 批量计算QKV
        kv = self.to_qkv(x).chunk(2, dim=-1)  # 只取K和V
        q = self.to_q_center(center_pixel)
        k, v = kv
        
        # 重塑为多头格式 [b, heads, n/1, dim_heads]
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)
        
        # 计算空间注意力权重
        # q_norm = F.normalize(q, dim=-1)
        # k_norm = F.normalize(k, dim=-1)
        q = q*self.scale
        attn_spatial = torch.einsum('bhqd,bhnd->bhqn', q, k) # + g_pos # b h 1 q
        # print(attn_spatial.shape)
        # exit()
        # 融合高斯位置先验
        # a = self.alpha.sigmoid()
        attention_weights = attn_spatial + self.alpha * g_pos # b h 1 n
        attention_weights = attention_weights.softmax(dim=-1).transpose(-1, -2) # b h n 1
        # 应用注意力权重 b h n d * b h n 1
        out = v * attention_weights # + v
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = out # + x  # 残差连接
        
        # 第二阶段：光谱注意力
        # 准备Q
        # q_self = self.q_lr_self(out[:, center_idx:center_idx+1, :]).expand(b, n, d)
        # q_self = rearrange(q_self, 'b n (h d) -> b h n d', h=h)
        
        # 准备Q（使用卷积处理空间特征）
        out_2d = rearrange(x, 'b (h1 w1) d -> b d h1 w1', h1=h1, w1=w1)
        q_self = self.q_lr_self(out_2d)  # [b, inner_dim, h1, w1]
        
        # 提取Q的中心像素
        center_h, center_w = h1 // 2, w1 // 2
        q_center = q_self[:, :, center_h, center_w]  # [b, inner_dim]
        q_center = q_center.unsqueeze(1)  # [b, 1, inner_dim]
        
        # 调整为多头格式：B h 1 d
        # q_center = q_center.view(b, 1, h, self.dim_heads).transpose(1, 2)  # [b, h, 1, dim_heads]
        q_center = rearrange(q_center, 'b n (h d) -> b h d n', h=h)
        
        # 准备K和V
        k_self = self.self_attn_k(out_2d)  # [b, inner_dim, h1, w1]
        v_self = self.self_attn_v(out_2d)  # [b, inner_dim, h1, w1]
        
        # 将k_self的空间像素平均得到B h 1 d
        k_self_avg = k_self.mean(dim=(2, 3)).unsqueeze(1)  # [b, inner_dim]
        k_self_avg = rearrange(k_self_avg, 'b n (h d) -> b h d n', h=h) # k_self_avg.view(b, 1, h, self.dim_heads).transpose(1, 2)  # [b, h, 1, dim_heads]
        
        # 交换Q的最后两个维度，然后与K相乘得到B h d d的光谱注意力
        # q_center_swapped = q_center.transpose(-1, -2)  # [b, h, dim_heads, 1]
        spectral_attn =  q_center@k_self_avg.transpose(-1, -2) # torch.einsum('bhdi,bhdj->bhij', q_center_swapped, k_self_avg)  # [b, h, dim_heads, dim_heads]
        
        # 添加DecayPos1d位置编码
        realPos = self.realPos(self.dim_heads)  # [heads, dim_heads, dim_heads]
        realPos = realPos.unsqueeze(0)  # [1, heads, dim_heads, dim_heads]
        spectral_attn = spectral_attn  + realPos
        # spectral_attn = spectral_attn * self.temperature + realPos
        spectral_attn = spectral_attn.softmax(dim=-1)
        
        # 准备V用于后续计算
        v_self = rearrange(v_self, 'b (hd h_d) h1 w1 -> b hd h_d (h1 w1) ', hd=h, h_d=self.dim_heads)
        
        # 将V和光谱注意力相乘，得到输出B h n d
        # out_spectral = torch.einsum('bhnd,bhdc->bhnc', v_self, spectral_attn.transpose(-1, -2))  # [b, h, n, dim_heads]
        out_spectral = spectral_attn @ v_self
        # 重新调整为b n d
        # out_final = out_spectral.transpose(1, 2).contiguous()  # [b, n, h, dim_heads]
        # out_final = out_final.view(b, n, self.inner_dim)   # [b, n, inner_dim]
        out_final = rearrange(out_spectral, 'b h d n -> b n (h d)')
        # out_final_ = out_final.softmax(dim=-1)*x + out.softmax(dim=-2)*x
        # out_final_ = torch.cat([out_final, out], dim=-1)
        out_final_ = out_final * out
        # out_final_ = out_final
        # 添加线性层和dropout层
        out_final = self.final_out(out_final_) 
        
        return out_final

class Transformer(nn.Module):
    """Transformer编码器"""
    def __init__(self, dim, depth, heads, dim_heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Residual(LayerNormalize(dim, CenterPixelAttention(dim, heads, dim_heads, dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout)))
            ]) for _ in range(depth)
        ])

    def forward(self, x, mask=None):
        x_center = []
        center_idx = x.shape[1] // 2
        
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)
            x = mlp(x)
            x_center.append(x[:, center_idx, :])
            
        return x, x_center

# 标准注意力
class StandardAttention(nn.Module):
    """
    标准QKV注意力机制，包含空间注意力和光谱注意力
    不使用中心像素提取
    """
    def __init__(self, dim, heads, dim_heads, dropout):
        super().__init__()
        inner_dim = dim_heads * heads
        self.heads = heads
        self.dim = dim
        self.inner_dim = inner_dim
        self.dim_heads = dim_heads
        self.scale = dim_heads**(-0.5)
        
        # 标准QKV映射
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        # 光谱注意力的映射
        self.q_lr_self = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.Conv2d(dim, inner_dim, kernel_size=1)
        )
        self.self_attn_k = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.Conv2d(dim, inner_dim, kernel_size=1)
        )
        self.self_attn_v = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.Conv2d(dim, inner_dim, kernel_size=1)
        )
        
        # 位置编码参数
        # self.realPos = DecayPos1d(dim, heads, 2, 4)
        # self.alpha = nn.Parameter(torch.ones(1)*0.01)
        # 输出层
        self.final_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        b, n, d = x.shape
        h = self.heads
        h1 = w1 = int(math.sqrt(n))
        
        # g_pos = gaussian_spatial_position(h1, w1, device=x.device)
        # g_pos = g_pos.view(-1).unsqueeze(0).unsqueeze(0).unsqueeze(0) # 1 1 1 n
        # 第一阶段：标准空间注意力（不使用中心像素）
        # 批量计算QKV
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = qkv
        
        # 重塑为多头格式 [b, heads, n, dim_heads]
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)
        
        # 计算标准空间注意力权重（不使用高斯位置）
        # q_norm = F.normalize(q, dim=-1)
        # k_norm = F.normalize(k, dim=-1)
        attn_spatial = torch.einsum('bhqd,bhnd->bhqn', q, k)*self.scale # + self.alpha*g_pos
        
        # 应用空间注意力权重
        out = attn_spatial.softmax(dim=-1) @ v
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # 第二阶段：标准光谱注意力（不使用中心像素）
        # 准备Q（使用卷积处理空间特征）
        out_2d = rearrange(x, 'b (h1 w1) d -> b d h1 w1', h1=h1, w1=w1)
        q_self = self.q_lr_self(out_2d)  # [b, inner_dim, h1, w1]
        
        # 准备K和V
        k_self = self.self_attn_k(out_2d)  # [b, inner_dim, h1, w1]
        v_self = self.self_attn_v(out_2d)  # [b, inner_dim, h1, w1]
        
        # 将Q、K、V都调整为多头格式：B h n d
        q_self = rearrange(q_self, 'b (hd h_d) h1 w1 -> b hd h_d (h1 w1)', hd=h, h_d=self.dim_heads)
        k_self = rearrange(k_self, 'b (hd h_d) h1 w1 -> b hd h_d (h1 w1)', hd=h, h_d=self.dim_heads)
        v_self = rearrange(v_self, 'b (hd h_d) h1 w1 -> b hd h_d (h1 w1)', hd=h, h_d=self.dim_heads)
        
        # 计算标准光谱注意力：B h d d
        # 使用标准的QK计算，不使用中心像素
        scale = math.sqrt(n)**(-0.5)
        spectral_attn = torch.einsum('bhdn,bhkn->bhdk', q_self*scale, k_self)  # [b, h, dim_heads, dim_heads]
        
        # 添加DecayPos1d位置编码
        # realPos = self.realPos(self.dim_heads)  # [heads, dim_heads, dim_heads]
        # realPos = realPos.unsqueeze(0)  # [1, heads, dim_heads, dim_heads]
        spectral_attn = spectral_attn  # + realPos
        spectral_attn = spectral_attn.softmax(dim=-1)
        # 将V和光谱注意力相乘，得到输出B h n d
        out_spectral = spectral_attn @ v_self # b hd h_d n
         
        # 重新调整为b n d
        # out_final = out_spectral.transpose(1, 2).contiguous()  # [b, n, h, dim_heads]
        # out_final = out_final.view(b, n, self.inner_dim)  # [b, n, inner_dim]
        out_spectral = rearrange(out_spectral, 'b h d n -> b n (h d)')
        # 融合空间注意力和光谱注意力的输出
        # out_combined = out.softmax(dim=-2)*x + out_final.softmax(dim=-1)*x  # 简单相加融合
        out_spectral = out_spectral*out
        # 添加线性层和dropout层
        out_final = self.final_out(out_spectral)
        
        return out_final

class CPFormer(nn.Module):
    """SQSFormer主模型"""
    def __init__(self, params):
        super().__init__()
        net_params = params['net']
        data_params = params['data']

        # 模型参数
        num_classes = data_params.get("num_classes", 16)
        self.spectral_size = data_params.get("spectral_size", 200)
        
        # 网络参数
        depth = net_params.get("depth", 1)
        heads = net_params.get("heads", 8)
        mlp_dim = net_params.get("mlp_dim", 8)
        kernal = net_params.get('kernal', 3)
        padding = net_params.get('padding', 1)
        dropout = net_params.get("dropout", 0)
        dim = net_params.get("dim", 64)
        conv2d_out = dim
        dim_heads = dim // heads
        
        
        self.conv1 = nn.Conv2d(self.spectral_size, self.spectral_size, kernel_size=1)
        self.conv2d_features = nn.Sequential(
            nn.Conv2d(self.spectral_size*2 , conv2d_out, kernel_size=kernal, padding=padding),
            nn.BatchNorm2d(conv2d_out),
            nn.ReLU(inplace=True)
        )
        
        self.local_trans_pixel = Transformer(dim, depth, heads, dim_heads, mlp_dim, dropout)
        
        linear_dim = dim * 2

        self.classifier_mlp = nn.Sequential(
            nn.Linear(dim, linear_dim),
            nn.BatchNorm1d(linear_dim),
            nn.Dropout(0.1),
            nn.ReLU(inplace=True),
            nn.Linear(linear_dim, num_classes)
        )

    def encoder_block(self, x):
        
        b, s, h, w = x.shape
        x = x.permute(0, 3, 1, 2)  
        
        
        x_conv1 = self.conv1(x)
        x_concat = torch.cat([x_conv1, x], dim=1)
        x_features = self.conv2d_features(x_concat)
        
        x_seq = rearrange(x_features, 'b c h w -> b (h w) c')
        
        x_encoded, _ = self.local_trans_pixel(x_seq)
        
        reduce_x = torch.mean(x_encoded, dim=1)
        
        return _, reduce_x

    def forward(self, x, left=None, right=None):
        _, logit_x = self.encoder_block(x)
        
        mean_left = mean_right = None
        if left is not None and right is not None:
            _, mean_left = self.encoder_block(left)
            _, mean_right = self.encoder_block(right)
        
        return self.classifier_mlp(logit_x), mean_left, mean_right


# ============= 使用示例 =============
if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 模型参数配置
    params = {
        'net': {
            'model_type': 0,
            'depth': 1,
            'heads': 8,
            'mlp_dim': 8,
            'kernel': 3,
            'padding': 1,
            'dropout': 0,
            'dim': 64
        },
        'data': {
            'num_classes': 16,
            'patch_size': 13,
            'spectral_size': 200
        }
    }
    
    # 创建模型
    model = CPFormer(params).to(device)
    
    # 测试输入
    batch_size = 4
    spectral_size = params['data']['spectral_size']
    patch_size = params['data']['patch_size']
    
    x = torch.randn(batch_size, patch_size, patch_size, spectral_size).to(device)
    # left = torch.randn(batch_size, spectral_size, patch_size, patch_size).to(device)
    # right = torch.randn(batch_size, spectral_size, patch_size, patch_size).to(device)
    left = right = None
    # 前向传播测试
    model.eval()
    with torch.no_grad():
        output, mean_left, mean_right = model(x, left, right)
    
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"左分支输出: {mean_left.shape if mean_left is not None else 'None'}")
    print(f"右分支输出: {mean_right.shape if mean_right is not None else 'None'}")
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数统计:")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 性能测试
    if torch.cuda.is_available():
        import time
        
        model.eval()
        torch.cuda.synchronize()
        
        # 预热
        for _ in range(10):
            with torch.no_grad():
                _ = model(x)
        
        # 计时
        torch.cuda.synchronize()
        start_time = time.time()
        
        iterations = 100
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(x)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations
        print(f"\n性能测试:")
        print(f"推理速度: {avg_time*1000:.2f} ms/batch")
        print(f"吞吐量: {batch_size/avg_time:.2f} samples/s")
    
    # 梯度健康检查
    model.train()
    output, _, _ = model(x)
    loss = output.mean()
    loss.backward()
    
    print("\n梯度健康检查:")
    grad_healthy = True
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm < 1e-8:
                print(f"⚠️  {name}: 梯度可能消失 (norm={grad_norm:.2e})")
                grad_healthy = False
            elif grad_norm > 100:
                print(f"⚠️  {name}: 梯度可能爆炸 (norm={grad_norm:.2e})")
                grad_healthy = False
    
    if grad_healthy:
        print("✓ 所有梯度正常")
    
    print("\n优化完成！")