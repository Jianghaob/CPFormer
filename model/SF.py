import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
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
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, mask = None):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout = dropout)))
            ]))

        self.mode = mode
        self.skipcat = nn.ModuleList([])
        for _ in range(depth-2):
            self.skipcat.append(nn.Conv2d(num_channel+1, num_channel+1, [1, 2], 1, 0))

    def forward(self, x, mask = None):
        if self.mode == 'ViT':
            for attn, ff in self.layers:
                x = attn(x, mask = mask)
                x = ff(x)
        elif self.mode == 'CAF':
            last_output = []
            nl = 0
            for attn, ff in self.layers:           
                last_output.append(x)
                if nl > 1:             
                    x = self.skipcat[nl-2](torch.cat([x.unsqueeze(3), last_output[nl-2].unsqueeze(3)], dim=3)).squeeze(3)
                x = attn(x, mask = mask)
                x = ff(x)
                nl += 1

        return x

class ViT(nn.Module):
    def __init__(self, image_size, near_band, num_patches, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=1, dim_head = 16, dropout=0., emb_dropout=0., mode='ViT'):
        super().__init__()

        patch_dim = image_size ** 2 * near_band
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches, mode)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    def forward(self, x, mask = None):
       
        # patchs[batch, patch_num, patch_size*patch_size*c]  [batch,200,145*145]
        # x = rearrange(x, 'b c h w -> b c (h w)')

        ## embedding every patch vector to embedding size: [batch, patch_num, embedding_size]
        x = self.patch_to_embedding(x) #[b,n,dim]
        b, n, _ = x.shape

        # add position embedding
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b) #[b,1,dim]
        x = torch.cat((cls_tokens, x), dim = 1) #[b,n+1,dim]
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # transformer: x[b,n + 1,dim] -> x[b,n + 1,dim]
        x = self.transformer(x, mask)

        # classification: using cls_token output
        x = self.to_latent(x[:,0])

        # MLP classification layer
        return self.mlp_head(x)


if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 模型参数配置
    image_size = 9  # 图像补丁大小 (例如 9x9)
    near_band = 3  # 近邻波段数
    num_patches = 200  # 补丁数量
    num_classes = 16  # 分类类别数
    dim = 128  # 嵌入维度
    depth = 4  # Transformer深度
    heads = 8  # 注意力头数
    mlp_dim = 256  # MLP隐藏层维度
    dim_head = 16  # 每个注意力头的维度
    dropout = 0.1  # Dropout率
    emb_dropout = 0.1  # 嵌入层Dropout率
    batch_size = 4  # 批次大小
    
    # 计算patch_dim
    patch_dim = image_size ** 2 * near_band
    print(f"\n模型参数:")
    print(f"  image_size: {image_size}")
    print(f"  near_band: {near_band}")
    print(f"  num_patches: {num_patches}")
    print(f"  patch_dim: {patch_dim} (= {image_size}^2 * {near_band})")
    print(f"  num_classes: {num_classes}")
    print(f"  dim: {dim}")
    print(f"  depth: {depth}")
    print(f"  heads: {heads}")
    
    # 创建模型 - ViT模式
    print("\n" + "="*50)
    print("测试 ViT 模式")
    print("="*50)
    model_vit = ViT(
        image_size=image_size,
        near_band=near_band,
        num_patches=num_patches,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dim_head=dim_head,
        dropout=dropout,
        emb_dropout=emb_dropout,
        mode='ViT'
    ).to(device)
    model_vit.eval()
    
    # 创建测试输入: [batch, num_patches, patch_dim]
    test_input = torch.randn(batch_size, num_patches, patch_dim).to(device)
    print(f"\n输入形状: {test_input.shape}")
    print(f"  期望: (batch={batch_size}, num_patches={num_patches}, patch_dim={patch_dim})")
    
    # 前向传播测试
    print("\n开始前向传播...")
    try:
        with torch.no_grad():
            output = model_vit(test_input)
        
        print(f"✓ 前向传播成功！")
        print(f"输出形状: {output.shape}")
        print(f"期望输出形状: ({batch_size}, {num_classes})")
        
        # 验证输出形状
        assert output.shape == (batch_size, num_classes), \
            f"输出形状不匹配！期望: ({batch_size}, {num_classes}), 实际: {output.shape}"
        print("✓ 输出形状验证通过！")
        
        # 参数统计
        total_params = sum(p.numel() for p in model_vit.parameters())
        trainable_params = sum(p.numel() for p in model_vit.parameters() if p.requires_grad)
        print(f"\n模型参数统计:")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数量: {trainable_params:,}")
        
        # 测试反向传播
        print("\n测试反向传播...")
        model_vit.train()
        test_input.requires_grad = True
        output = model_vit(test_input)
        loss = output.sum()
        loss.backward()
        print("✓ 反向传播成功！")
        
    except Exception as e:
        print(f"\n❌ ViT模式测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试 CAF 模式
    print("\n" + "="*50)
    print("测试 CAF 模式")
    print("="*50)
    try:
        model_caf = ViT(
            image_size=image_size,
            near_band=near_band,
            num_patches=num_patches,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
            mode='CAF'
        ).to(device)
        model_caf.eval()
        
        test_input_caf = torch.randn(batch_size, num_patches, patch_dim).to(device) # #############################################
        print(f"\n输入形状: {test_input_caf.shape}")
        
        with torch.no_grad():
            output_caf = model_caf(test_input_caf)
        
        print(f"✓ CAF模式前向传播成功！")
        print(f"输出形状: {output_caf.shape}")
        assert output_caf.shape == (batch_size, num_classes), \
            f"CAF模式输出形状不匹配！期望: ({batch_size}, {num_classes}), 实际: {output_caf.shape}"
        print("✓ CAF模式输出形状验证通过！")
        
    except Exception as e:
        print(f"\n❌ CAF模式测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试不同输入尺寸
    print("\n" + "="*50)
    print("测试不同输入尺寸")
    print("="*50)
    try:
        model_vit.eval()
        test_sizes = [
            (2, 100, patch_dim),  # 不同batch size和patch数量
            (1, num_patches, patch_dim),  # batch size = 1
        ]
        
        for i, (b, n, p) in enumerate(test_sizes):
            test_input_var = torch.randn(b, n, p).to(device)
            # 需要重新创建模型以匹配新的patch数量
            model_var = ViT(
                image_size=image_size,
                near_band=near_band,
                num_patches=n,
                num_classes=num_classes,
                dim=dim,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
                dim_head=dim_head,
                dropout=dropout,
                emb_dropout=emb_dropout,
                mode='ViT'
            ).to(device)
            model_var.eval()
            
            with torch.no_grad():
                output_var = model_var(test_input_var)
            
            assert output_var.shape == (b, num_classes), \
                f"测试 {i+1} 输出形状错误: {output_var.shape}"
            print(f"  ✓ 测试 {i+1}: batch={b}, num_patches={n} -> 输出形状 {output_var.shape}")
        
    except Exception as e:
        print(f"\n❌ 不同输入尺寸测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*50)
    print("所有测试完成！✓")
    print("="*50)

