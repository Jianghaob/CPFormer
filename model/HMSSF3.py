import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init
import math
# from keras.layers import Lambda, Concatenate, Add, Softmax, Layer
# import keras.backend as K
# from config import config

pca_components = 30
# 3DConv parameters
dim_3Dout = 5
dim_3DKernel1 = 3
dim_3DKernel23 = 3
#spactial branch
dim1 = 60 # 编码器1的嵌入维度
dim2 = 60 # 
dim3 = 120
dim_linear = pca_components - dim_3DKernel1 + 1
patch_size = 15
dim_patch = patch_size - dim_3DKernel23 +1
# spactial branch
dim1_SPE = dim1
dim2_SPE = dim2
dim3_SPE = dim3
# common parameters
dim_classes = 240

def max_factor(num):
    factors = []
    for i in range(1, int(num**0.5) + 1):
        if num % i == 0:
            factors.append(i)
    return torch.max(torch.tensor(factors))

class Attention_FeaMix(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.q = nn.Linear(size, 1, bias='True')

    def forward(self, x):
        stream1, stream2 = x[0], x[1]

        d1 = self.q(stream1)  #[64, 1]
        d2 = self.q(stream2)  #[64, 1]
        ds = torch.cat([d1, d2], dim=1)

        # d1 and d2 and of size (bs, 1) individually
        # ds of size (bs, 2)

        tmp = ds.softmax(dim=1)
        # print(tmp._keras_shape)
        w1 = tmp[:, 0].unsqueeze(dim=1)
        w2 = tmp[:, 1].unsqueeze(dim=1)

        # print(w1._keras_shape)
        # print(w1.shape)
        # print(w2.shape)

        stream1 = w1 * stream1
        stream2 = w2 * stream2
        # result = torch.cat([stream1, stream2], dim=-1)
        result = stream1 + stream2
        # print(result.shape)
        return result

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class MLP_Block(nn.Module):
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

class Attention(nn.Module):
    def __init__(self, dim, heads=None, dropout=0.1, qkv_bias=True, attn_drop=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)

        self.nn1 = nn.Linear(dim, dim)
        self.do1 = nn.Dropout(dropout)
        self.sr = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

        self.num_heads = 3
        self.K_SPA = [1, 2, 3]
        self.S = [1, 1, 1]
        self.split = len(self.K_SPA)
        self.sr1 = nn.Conv2d(dim, dim, kernel_size=self.K_SPA[0], stride=self.S[0])
        self.norm1 = nn.LayerNorm(dim)
        self.sr2 = nn.Conv2d(dim, dim, kernel_size=self.K_SPA[1], stride=self.S[1])
        self.norm2 = nn.LayerNorm(dim)
        self.sr3 = nn.Conv2d(dim, dim, kernel_size=self.K_SPA[2], stride=self.S[2])
        self.norm3 = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.local_conv1 = nn.Conv2d(dim//self.num_heads, dim//self.num_heads, kernel_size=3, padding=1, stride=1, groups=dim//self.num_heads)
        self.local_conv2 = nn.Conv2d(dim//self.num_heads, dim//self.num_heads, kernel_size=3, padding=1, stride=1, groups=dim//self.num_heads)
        self.local_conv3 = nn.Conv2d(dim//self.num_heads, dim//self.num_heads, kernel_size=3, padding=1, stride=1, groups=dim//self.num_heads)
        self.attn_drop = nn.Dropout(attn_drop)

        self.K_SPE = [1, 5, 10]    #[1, 3, 6]
        self.sr1_SPE = nn.Conv3d(1, out_channels=1, kernel_size=(self.K_SPE[0], 1, 1), stride=(1, 1, 1))
        self.sr2_SPE = nn.Conv3d(1, out_channels=1, kernel_size=(self.K_SPE[1], 1, 1), stride=(1, 1, 1))
        self.sr3_SPE = nn.Conv3d(1, out_channels=1, kernel_size=(self.K_SPE[2], 1, 1), stride=(1, 1, 1))
        self.local_conv1_SPE = nn.Conv3d(1, out_channels=1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.local_conv2_SPE = nn.Conv3d(1, out_channels=1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.local_conv3_SPE = nn.Conv3d(1, out_channels=1, kernel_size=(1, 1, 1), stride=(1, 1, 1))

    def forward(self, x, mask=None, use_SR=False):
        b, n, d, h = *x.shape, self.heads
        H = int(math.sqrt(n))
        W = int(math.sqrt(n))

        # sr k, v
        if mask=='Shunted':
            B, N, C = x.shape # 64 169 60
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # [64, 3, 169, 20]
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)  # [64, 169, 60]->[64, 60, 13, 13]
            x_1 = self.act(self.norm1(self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1)))  # 64 169 60
            x_2 = self.act(self.norm2(self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1)))  #64 100 60 [64, 48, 8, 8]->[64, 48, 6, 6]->[64, 48, 36]->[64, 36, 48]
            x_3 = self.act(self.norm3(self.sr3(x_).reshape(B, C, -1).permute(0, 2, 1)))  # 64 49 60[64, 48, 8, 8]->[64, 48, 3, 3]->[64, 48, 9]->[64, 9, 48]
            kv1 = self.kv1(x_1).reshape(B, -1, self.num_heads, 1, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 64 169 3 1 30 -> 3 64 1 169 30 [64, 64, 48]->[64, 64, 48]->[64, 64, 3, 1, 16]->[3, 64, 1, 64, 16]
            kv2 = self.kv1(x_2).reshape(B, -1, self.num_heads, 1, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 64 100 3 1 30 -> 3 64 1 100 30
            kv3 = self.kv1(x_3).reshape(B, -1, self.num_heads, 1, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 64 49 3 1 30 -> 3 64 1 49 30
            k1, v1 = kv1[0], kv1[1]  # 64 1 169 20
            k2, v2 = kv2[0], kv2[1]  # 64 1 100 20
            k3, v3 = kv3[0], kv3[1]  # 64 1 49 20
            attn1 = (q[:, :1] @ k1.transpose(-2, -1)) * self.scale  # [64, 1, 169, 20] * [64, 1, 20, 169] = [64, 3, 169, 169] @ 64 1 169 20
            attn1 = self.attn_drop(attn1) 
            # 64 1 169 20,给v加卷积
            v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C//self.num_heads).transpose(1, 2).view(B, C//self.num_heads, int(math.sqrt(v1.shape[2])), -1)).\
                view(B, C//self.num_heads, -1).view(B, 1, C // self.num_heads, -1).transpose(-1, -2) #[64, 1, 64, 16]
            #[64, 1, 64, 16]->[64, 64, 1, 16]->[64, 64, 16]->[64, 16, 64]->[64, 24, 8, 8]->conv1[64, 24, 8, 8]->[64, 24, 64])->[64, 1, 24, 64]->[64, 1, 64, 24]
            x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C//self.num_heads) # 64 169 20  #attn1[64, 1, 64, 64] * v1[64, 1, 64, 16] = x1[64, 1, 64, 16]->[64, 64, 1, 16]->[64, 64, 16]
            attn2 = (q[:, 1:2] @ k2.transpose(-2, -1)) * self.scale  #q[64, 1, 64, 16] * k[64, 1, 16, 36] = attn[64, 1, 64, 36]
            attn2 = attn2.softmax(dim=-1)
            attn2 = self.attn_drop(attn2)
            v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C//self.num_heads).transpose(1, 2).view(B, C//self.num_heads, int(math.sqrt(v2.shape[2])), -1)).\
                view(B, C//self.num_heads, -1).view(B, 1, C // self.num_heads, -1).transpose(-1, -2)  #[64, 1, 36, 16]
            #[64, 1, 36, 16]->[64, 36, 1, 16]->[64, 36, 16]->[64, 16, 36]->[64, 16, 6, 6]->conv2[64, 16, 6, 6]->[64, 16, 36])->[64, 1, 16, 36]->[64, 1, 36, 16]
            # 64 100 20
            x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C//self.num_heads)  #attn2[64, 1, 64, 36] * v2[64, 1, 36, 16] = x2[64, 1, 64, 16]->[64, 64, 1, 16]->[64, 64, 16]
            attn3 = (q[:, 2:3] @ k3.transpose(-2, -1)) * self.scale  #q[64, 1, 64, 16] * k[64, 1, 16, 9] = attn[64, 1, 64, 9]
            attn3 = attn3.softmax(dim=-1)
            attn3 = self.attn_drop(attn3)
            # 64 49 20
            v3 = v3 + self.local_conv3(v3.transpose(1, 2).reshape(B, -1, C//self.num_heads).transpose(1, 2).view(B, C//self.num_heads, int(math.sqrt(v3.shape[2])), -1)).\
                view(B, C//self.num_heads, -1).view(B, 1, C // self.num_heads, -1).transpose(-1, -2)  #[64, 1, 9, 16]
            #[64, 1, 9, 16]->[64, 9, 1, 16]->[64, 9, 16]->[64, 16, 9]->[64, 16, 3, 3]->conv2[64, 16, 3, 3]->[64, 16, 9])->[64, 1, 16, 9]->[64, 1, 9, 16]
            x3 = (attn3 @ v3).transpose(1, 2).reshape(B, N, C//self.num_heads)  #attn3[64, 1, 64, 9] * v3[64, 1, 9, 16] = x3[64, 1, 64, 16]->[64, 64, 1, 16]->[64, 64, 16]

            out = torch.cat([x1, x2, x3], dim=-1)  #[64, 169, 60]
        elif mask=='Shunted_SPE':
            B, N, C = x.shape
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # [64, 3, 84, 27]
            x_ = x.reshape(B, N, max_factor(x.shape[2]), -1).unsqueeze(dim=1)  #[64, 84, 48] --> [64, 1, 84, 6, 8]
            x_1 = self.act(self.norm1(self.sr1_SPE(x_).reshape(B, C, -1).permute(0, 2, 1)))  #[64, 84, 144]
            x_2 = self.act(self.norm2(self.sr2_SPE(x_).reshape(B, C, -1).permute(0, 2, 1)))  #[64, 57, 144]
            x_3 = self.act(self.norm3(self.sr3_SPE(x_).reshape(B, C, -1).permute(0, 2, 1)))  #[64, 71, 144]
            kv1 = self.kv1(x_1).reshape(B, -1, self.num_heads, 1, C // self.num_heads).permute(2, 0, 3, 1, 4)  # [3, 64, 1, 84, 48]
            kv2 = self.kv1(x_2).reshape(B, -1, self.num_heads, 1, C // self.num_heads).permute(2, 0, 3, 1, 4)  # [3, 64, 1, 57, 48]
            kv3 = self.kv1(x_3).reshape(B, -1, self.num_heads, 1, C // self.num_heads).permute(2, 0, 3, 1, 4)  # [3, 64, 1, 71, 48]
            k1, v1 = kv1[0], kv1[1]  # [64, 1, 84, 48]
            k2, v2 = kv2[0], kv2[1]  # [64, 1, 57, 48]
            k3, v3 = kv3[0], kv3[1]  # [64, 1, 71, 48]
            attn1 = (q[:, :1] @ k1.transpose(-2, -1)) * self.scale  # q[64, 1, 84, 48] * k[64, 1, 48, 84]= attn[64, 1, 84, 84]
            attn1 = attn1.softmax(dim=-1)
            attn1 = self.attn_drop(attn1)
            v1 = v1 + self.local_conv1_SPE(v1.unsqueeze(dim=4)).squeeze(dim=4) #[64, 1, 84, 48]
            x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C//self.num_heads) #[64, 84, 48]
            attn2 = (q[:, 1:2] @ k2.transpose(-2, -1)) * self.scale  #q[64, 1, 84, 48] * k[64, 1, 48, 57] = attn[64, 1, 84, 57]
            attn2 = attn2.softmax(dim=-1)
            attn2 = self.attn_drop(attn2)
            v2 = v2 + self.local_conv2_SPE(v2.unsqueeze(dim=4)).squeeze(dim=4) #[64, 1, 57, 48]
            x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C//self.num_heads)  #attn2[64, 1, 84, 57] * v2[64, 1, 57, 48] = x2[64, 1, 84, 48]->[64, 84, 48]
            attn3 = (q[:, 2:3] @ k3.transpose(-2, -1)) * self.scale  #q[64, 1, 84, 48] * k[64, 1, 48, 71] = attn[64, 1, 84, 71]
            attn3 = attn3.softmax(dim=-1)
            attn3 = self.attn_drop(attn3)
            v3 = v3 + self.local_conv3_SPE(v3.unsqueeze(dim=4)).squeeze(dim=4) #[64, 1, 71, 48]
            x3 = (attn3 @ v3).transpose(1, 2).reshape(B, N, C//self.num_heads) #q[64, 1, 84, 71] * v[64, 1, 71, 48] = x[64, 84, 48]

            out = torch.cat([x1, x2, x3], dim=-1)  #[2, 3136, 64]
        else:
            qkv = self.to_qkv(x).chunk(3, dim = -1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split into multi head attentions
            #q[64, 8, 65, 4]  [64, 8, 17, 8]  [64, 8, 5, 16]
            dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale   #[64, 8, 65, 65]
            attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper
            out = torch.einsum('bhij,bhjd->bhid', attn, v)  #[64, 8, 65, 4] product of v times whatever inside softmax
            out = rearrange(out, 'b h n d -> b n (h d)')  # [64, 65, 32] concat heads into one matrix, ready for next encoder block

        out = self.nn1(out)   #[64, 64, 48]
        out = self.do1(out)   #[64, 64, 48]
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)
            x = mlp(x)
        return x

class LSFAT(nn.Module):
    def __init__(self, in_channels=1, num_classes=16, num_tokens=4, dim=64, depth=2, heads=1, mlp_dim=128, dropout=0.1, emb_dropout=0.1):
        super(LSFAT, self).__init__()
        self.L = num_tokens

        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels= dim_3Dout, kernel_size=(dim_3DKernel1, dim_3DKernel23, dim_3DKernel23)),
            nn.BatchNorm3d(dim_3Dout),
            nn.ReLU(),
        )

        self.spa_downks = [4, 1]
        self.size_patch2 = ((dim_patch - self.spa_downks[0]) // self.spa_downks[1]) + 1 # 13 - 4 = 9 + 1 = 10
        self.size_patch3 = ((self.size_patch2 - self.spa_downks[0]) // self.spa_downks[1]) + 1 # 10 - 4 = 6 + 1 = 7
        self.spe_downks = [4, 2]
        self.size_channel2 = ((dim_3Dout * dim_linear - self.spe_downks[0]) // self.spe_downks[1]) + 1 # 5*28 - 4 = 136 - 4 = 132 // 2 + 1 = 67
        self.size_channel3 = ((self.size_channel2 - self.spe_downks[0]) // self.spe_downks[1]) + 1 # 67 - 4 = 63 // 2 + 1 = 33

        self.patch_to_embedding1 = nn.Sequential(
            nn.Linear(dim_3Dout*dim_linear, dim1), # 5*28
            nn.LayerNorm(dim1),
        )
        self.patch_to_embedding2 = nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.LayerNorm(dim2),
        )
        self.patch_to_embedding3 = nn.Sequential(
            nn.Linear(dim2, dim3),
            nn.LayerNorm(dim3),
        )
        self.patch_to_embedding1_SPE = nn.Sequential(
            nn.Linear(dim_patch * dim_patch, dim1_SPE), # 169 -> 60
            nn.LayerNorm(dim1_SPE),
        )
        self.patch_to_embedding2_SPE = nn.Sequential(
            nn.Linear(dim1_SPE, dim2_SPE),
            nn.LayerNorm(dim2_SPE),
        )
        self.patch_to_embedding3_SPE = nn.Sequential(
            nn.Linear(dim2_SPE, dim3_SPE),
            nn.LayerNorm(dim3_SPE),
        )

        self.pos_embedding1 = nn.Parameter(torch.empty(1, dim_patch * dim_patch, dim1))
        torch.nn.init.normal_(self.pos_embedding1, std=.02)

        self.pos_embedding2 = nn.Parameter(torch.empty(1, self.size_patch2 * self.size_patch2, dim2))  #dim_patch//2 * dim_patch//2
        torch.nn.init.normal_(self.pos_embedding2, std=.02)

        self.pos_embedding3 = nn.Parameter(torch.empty(1, self.size_patch3 * self.size_patch3, dim3))  #dim_patch//4 * dim_patch//4
        torch.nn.init.normal_(self.pos_embedding3, std=.02)

        self.pos_embedding1_SPE = nn.Parameter(torch.empty(1, dim_3Dout * dim_linear, dim1_SPE))
        torch.nn.init.normal_(self.pos_embedding1_SPE, std=.02)

        self.pos_embedding2_SPE = nn.Parameter(torch.empty(1, self.size_channel2, dim2_SPE))
        torch.nn.init.normal_(self.pos_embedding2_SPE, std=.02)

        self.pos_embedding3_SPE = nn.Parameter(torch.empty(1, self.size_channel3, dim3_SPE))
        torch.nn.init.normal_(self.pos_embedding3_SPE, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim1))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer1 = Transformer(dim1, depth, heads, 64, dropout)
        self.transformer2 = Transformer(dim2, depth, heads, 16, dropout)
        self.transformer3 = Transformer(dim3, depth, heads, 4, dropout)

        self.transformer1_SPE = Transformer(dim1_SPE, depth, heads, 64, dropout)
        self.transformer2_SPE = Transformer(dim2_SPE, depth, heads, 16, dropout)
        self.transformer3_SPE = Transformer(dim3_SPE, depth, heads, 4, dropout)

        self.to_cls_token = nn.Identity()

        self.nn3 = nn.Linear(dim3, dim_classes)
        self.nn3_SPE = nn.Linear(dim3_SPE, dim_classes)
        self.Attention_FeaMix = Attention_FeaMix(dim_classes)
        self.nn = nn.Linear(dim_classes, num_classes)  # dim3 + dim3_SPE

    def LSFAT_Layer1(self, x, mask=None):
        x = rearrange(x, 'b c h w -> b (h w) c')   #64 169 140
        # pixel embedding
        x = self.patch_to_embedding1(x)  #64 169 60
        # cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  #[64, 1, 32]
        # x = torch.cat((cls_tokens, x), dim=1) #[64, 65, 32]
        x += self.pos_embedding1  #self.pos_embedding1[1, 144, 48]    x[64, 144, 48]
        x = self.dropout(x)
        # neighborhood aggregation attention
        x = self.transformer1(x, mask) #[64, 144, 48]
        # separate cls token and feature token
        # c = self.to_cls_token(x[:, 0]) # cls token [64, 32]
        # x = self.to_cls_token(x[:, 1:]) # feature token [64, 64, 32]
        return x

    def LSFAT_Layer2(self, p, c=None, mask=None, dim=dim1, k=0):
        p = p.reshape(p.shape[0], int(math.sqrt(p.shape[1])), -1, dim)  # 64 13 13 60
        # 使用平均池化替代双重循环：窗口大小4，步长1
        # 将 [B, H, W, dim] 转换为 [B, dim, H, W] 以便使用 avg_pool2d
        p_2d = p.permute(0, 3, 1, 2)  # [B, dim, H, W]
        # 使用平均池化：kernel_size=4, stride=1
        x_pooled = F.avg_pool2d(p_2d, kernel_size=self.spa_downks[0], stride=self.spa_downks[1])  # [B, dim, size_patch2, size_patch2]
        # 转换回 [B, size_patch2*size_patch2, dim]
        x = x_pooled.permute(0, 2, 3, 1).reshape(p.shape[0], self.size_patch2 * self.size_patch2, dim)  # [B, size_patch2^2, dim]
        
        # ========== 原始循环代码（已注释，保留供参考） ==========
        # x = torch.zeros(p.shape[0], self.size_patch2 * self.size_patch2, dim).cuda()  # 64 100 60
        # # neighborhood aggregation-based embedding 对p进行池化得到 64 13 13 60 -> 64 100 60
        # for i in range(0, self.size_patch2):
        #     for j in range(0, self.size_patch2):
        #         temp = p[:, i*self.spa_downks[1] : i*self.spa_downks[1]+self.spa_downks[0], j*self.spa_downks[1] : j*self.spa_downks[1]+self.spa_downks[0], :]  #[64, 2, 2, 32]
        #         temp = temp.reshape(temp.shape[0], -1, dim)  #[64, 4, 32]
        #         temp = temp.mean(dim=1)  #[64, 60]
        #         x[:,k,:] = temp # 
        #         k += 1
        # ====================================================
        
        x = self.patch_to_embedding2(x)  #[64, 16, 32]  -> [64, 16, 64]
        # c = self.patch_to_embedding2(c)  #[64, 32]  ->[64, 64]
        # cls_tokens = c.reshape(x.shape[0],1,dim2)

        # x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding2
        x = self.dropout(x)
        # neighborhood aggregation attention
        x = self.transformer2(x, mask)  #[64, 100, 60]
        # separate cls token and feature token
        # c = self.to_cls_token(x[:, 0]) # cls token [64, 64]
        # x = self.to_cls_token(x[:, 1:]) # feature token  [64, 16, 64]
        return x

    def LSFAT_Layer3(self, p, c=None, mask=None, dim=dim2, k=0):
        p = p.reshape(p.shape[0], int(math.sqrt(p.shape[1])), -1, dim)  # 64 100 60 -> 64 10 10 60
        # 使用平均池化替代双重循环：窗口大小4，步长1
        # 将 [B, H, W, dim] 转换为 [B, dim, H, W] 以便使用 avg_pool2d
        p_2d = p.permute(0, 3, 1, 2)  # [B, dim, H, W]
        # 使用平均池化：kernel_size=4, stride=1
        x_pooled = F.avg_pool2d(p_2d, kernel_size=self.spa_downks[0], stride=self.spa_downks[1])  # [B, dim, size_patch3, size_patch3]
        # 转换回 [B, size_patch3*size_patch3, dim]
        x = x_pooled.permute(0, 2, 3, 1).reshape(p.shape[0], self.size_patch3 * self.size_patch3, dim)  # [B, size_patch3^2, dim]
        
        # ========== 原始循环代码（已注释，保留供参考） ==========
        # x = torch.zeros(p.shape[0], self.size_patch3 * self.size_patch3, dim).cuda()  # 64 49 60
        # # neighborhood aggregation-based embedding
        # for i in range(0,self.size_patch3):
        #     for j in range(0,self.size_patch3):
        #         temp = p[:, i*self.spa_downks[1] : i*self.spa_downks[1]+self.spa_downks[0], j*self.spa_downks[1] : j*self.spa_downks[1]+self.spa_downks[0], :]  #[64, 2, 2, 32]
        #         temp = temp.reshape(temp.shape[0], -1, dim)  #[64, 4, 32]
        #         temp = temp.mean(dim=1)  #[64, 32]
        #         x[:,k,:] = temp
        #         k += 1
        # ====================================================
        
        x = self.patch_to_embedding3(x)   #[64, 49, 60]->[64, 49, 120]
        # c = self.patch_to_embedding3(c)   #[64, 64]->[64, 128]
        # cls_tokens = c.reshape(x.shape[0],1,dim3)

        # x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding3
        x = self.dropout(x)
        x = self.transformer3(x, mask)  #[64, 49, 120]

        # separate cls token and feature token
        # c = self.to_cls_token(x[:, 0]) # cls token [64, 128]
        # x = self.to_cls_token(x[:, 1:]) # feature token  #[64, 4, 128]
        return x

    def SPE_Layer1(self, x, mask=None):
        x = rearrange(x, 'b c h w -> b c (h w)')   # 64 140 13 13 -> 64 140 169
        # pixel embedding
        x = self.patch_to_embedding1_SPE(x)  #[64, 140, 60]
        x += self.pos_embedding1_SPE
        x = self.dropout(x)
        # neighborhood aggregation attention
        x = self.transformer1_SPE(x, mask) # [64, 140, 60] -> [64, 140, 60]
        return x

    def SPE_Layer2(self, p, c=None, mask=None, dim=dim1_SPE, k=0):
        # 使用1D平均池化替代循环：窗口大小4，步长2
        # 将 [B, N, dim] 转换为 [B, dim, N] 以便使用 avg_pool1d
        p_1d = p.permute(0, 2, 1)  # [B, dim, N]
        # 使用1D平均池化：kernel_size=4, stride=2
        x_pooled = F.avg_pool1d(p_1d, kernel_size=self.spe_downks[0], stride=self.spe_downks[1])  # [B, dim, size_channel2]
        # 转换回 [B, size_channel2, dim]
        x = x_pooled.permute(0, 2, 1)  # [B, size_channel2, dim]
        
        # ========== 原始循环代码（已注释，保留供参考） ==========
        # x = torch.zeros(p.shape[0], self.size_channel2, dim).cuda()  #64 67 60
        # # neighborhood aggregation-based embedding
        # for i in range(0, self.size_channel2): # p[:, i*2:i*2+4, :] -> [64, 2, 60]
        #     temp = p[:, i*self.spe_downks[1]:i*self.spe_downks[1]+self.spe_downks[0], :]  #[64, 2, 60]
        #     temp = temp.mean(dim=1)  #[64, 60]
        #     x[:,k,:] = temp
        #     k += 1
        # ====================================================
        
        x = self.patch_to_embedding2_SPE(x)  #[64, 67, 60]  -> [64, 67, 60]
        x += self.pos_embedding2_SPE
        x = self.dropout(x)
        # neighborhood aggregation attention
        x = self.transformer2_SPE(x, mask)  #[64, 67, 60] -> [64, 67, 60]
        # separate cls token and feature token
        # c = self.to_cls_token(x[:, 0]) # cls token [64, 64]
        # x = self.to_cls_token(x[:, 1:]) # feature token  [64, 16, 64]
        return x

    def SPE_Layer3(self, p, c=None, mask=None, dim=dim2_SPE, k=0):
        # 使用1D平均池化替代循环：窗口大小4，步长2
        # 将 [B, N, dim] 转换为 [B, dim, N] 以便使用 avg_pool1d
        p_1d = p.permute(0, 2, 1)  # [B, dim, N]
        # 使用1D平均池化：kernel_size=4, stride=2
        x_pooled = F.avg_pool1d(p_1d, kernel_size=self.spe_downks[0], stride=self.spe_downks[1])  # [B, dim, size_channel3]
        # 转换回 [B, size_channel3, dim]
        x = x_pooled.permute(0, 2, 1)  # [B, size_channel3, dim]
        
        # ========== 原始循环代码（已注释，保留供参考） ==========
        # x = torch.zeros(p.shape[0], self.size_channel3, dim).cuda()  # [64, 84, 48]
        # # neighborhood aggregation-based embedding
        # for i in range(0, self.size_channel3):
        #     temp = p[:, i*self.spe_downks[1]:i*self.spe_downks[1]+self.spe_downks[0], :]  # [64, 2, 48]
        #     temp = temp.mean(dim=1)  # [64, 48]
        #     x[:, k, :] = temp
        #     k += 1
        # ====================================================
        
        x = self.patch_to_embedding3_SPE(x)  #[64, 16, 32]  -> [64, 16, 64]
        # c = self.patch_to_embedding2(c)  #[64, 32]  ->[64, 64]
        # cls_tokens = c.reshape(x.shape[0],1,dim2)

        # x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding3_SPE
        x = self.dropout(x)
        # neighborhood aggregation attention
        x = self.transformer3_SPE(x, mask)  #[64, 17, 64]
        # separate cls token and feature token
        # c = self.to_cls_token(x[:, 0]) # cls token [64, 64]
        # x = self.to_cls_token(x[:, 1:]) # feature token  [64, 16, 64]
        return x

    def forward(self, img): # 64 1 30 15 15
        # print(img.shape)
        img = img.permute(0,1,4,2,3)
        # print(img.shape)
        # # 3d convolution
        img_3D = self.conv3d_features(img)   #[64, 1, 30, 15, 15]--> [64, 5, 28, 13, 13]
        img_3D = rearrange(img_3D, 'b c h w y -> b (c h) w y')   #[64, 140, 13, 13]
        # three-layer transformer
        x = self.LSFAT_Layer1(img_3D, mask='Shunted')   #[64, 144, 48]
        x = self.LSFAT_Layer2(x, mask='Shunted')  #[64, 100, 60]
        x = self.LSFAT_Layer3(x, mask='Shunted')  #[64, 49, 120]
        fea_spa = x.mean(dim=1)  # [64, 120]
        # pred = self.nn3(fea)  # [64, 9]

        # Y = self.conv3d_features_SE(x)  #[64, 1, 30, 15, 15]--> [64, 3, 28, 12, 12]
        # Y = rearrange(Y, 'b c h w y -> b (c h) w y')  # [64, 84, 12, 12]
        Y = self.SPE_Layer1(img_3D, mask='Shunted_SPE')   # [64, 140, 60]
        Y = self.SPE_Layer2(Y, mask='Shunted_SPE')  # [64, 67, 60]
        Y = self.SPE_Layer3(Y, mask='Shunted_SPE')  # [64, 33, 60]
        fea_spe = Y.mean(dim=1)  # [64, 60]
        # pred = self.nn3_SPE(fea)  # [64, 9]

        fea_spa = self.nn3(fea_spa)
        fea_spe = self.nn3_SPE(fea_spe)
        fea = self.Attention_FeaMix([fea_spa, fea_spe])  #[64, 120] or [64, 60]
        # fea = torch.cat([fea_spa, fea_spe], dim=-1) #[64, 288]
        # fea = fea_spa + fea_spe  # [64, 144]
        pred = self.nn(fea) #[64, 9]

        return pred # , fea

if __name__ == '__main__':
    model = LSFAT()
    device = torch.device("cuda:0")
    model = model.to(device)
    model.eval()
    print(model)
    input = torch.randn(64, 1, 15, 15, 30).cuda()
    y = model(input)
    print(y.size())
    