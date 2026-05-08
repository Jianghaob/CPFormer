import os
# 设置CUDA可见设备（如果环境变量未设置，则使用默认值）
# 在Docker容器中使用 --gpus all 时，可以不设置此变量，让系统自动分配
import time
import torch
import torch.nn as nn
import random
import shutil
import numpy as np
import gc  # 用于垃圾回收
from sklearn import metrics
from ptflops import get_model_complexity_info
try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False
    print("警告: fvcore未安装，多输入模型的FLOPs将无法计算")

import utils.evaluation as evaluation
import utils.data_load_operate as data_load_operate
import visual.cls_visual as cls_visual
import model.CPFormer as CPFormer
import model.CVSSN as CVSSN
import model.SQSFormer as SQSFormer
import model.DBDA as DBDA
import model.SSFTT as SSFTT
import model.SF as SFModel
import model.s2ftnet as S2FTNet
import model.HMSSF3 as HMSSF3
import model.CWSST as CWSST
from config.config import EXPERIMENT_CONFIG, MODEL_SPECS
# import utils.data_load_operate_AIPS as data_load_operate


time_current = time.strftime("%y-%m-%d-%H.%M", time.localtime())

# random seed setting
seed = 20

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# 清理所有GPU的显存缓存，解决多次中断导致的显存碎片化问题
def clear_gpu_cache():
    """
    清理所有GPU的显存缓存，释放碎片化的显存
    这有助于解决多次Ctrl+C中断导致的显存碎片化问题
    
    清理步骤：
    1. Python垃圾回收
    2. 清理所有GPU的PyTorch缓存
    3. 同步所有GPU操作
    """
    if torch.cuda.is_available():
        # 步骤1: Python垃圾回收，释放未引用的对象
        gc.collect()
        
        # 步骤2: 清理所有GPU的显存缓存
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()  # 清理进程间通信缓存
                torch.cuda.synchronize()  # 同步以确保清理完成
        
        print("已清理所有GPU的显存缓存（包括垃圾回收和碎片整理）")

# 设备选择逻辑：自动选择剩余可用显存最大的GPU
def select_best_gpu():
    """
    自动选择剩余可用显存最大的GPU
    返回: torch.device 对象
    """
    if not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        return torch.device("cpu")
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("未检测到GPU，使用CPU")
        return torch.device("cpu")
    
    if num_gpus == 1:
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        free_memory, total_memory = torch.cuda.mem_get_info(0)
        free_memory_gb = free_memory / (1024**3)
        total_memory_gb = total_memory / (1024**3)
        used_memory_gb = total_memory_gb - free_memory_gb
        usage_ratio = used_memory_gb / total_memory_gb if total_memory_gb > 0 else 0
        usage_warning = ""
        if usage_ratio > 0.9:
            usage_warning = " ⚠️ 显存使用率过高，建议重启Python进程清理显存"
        elif usage_ratio > 0.7:
            usage_warning = " ⚠️ 显存使用率较高"
        print(f"检测到1个GPU: {gpu_name}")
        print(f"  总显存: {total_memory_gb:.2f} GB, 已用: {used_memory_gb:.2f} GB ({usage_ratio*100:.1f}%), 剩余: {free_memory_gb:.2f} GB{usage_warning}")
        print(f"  使用设备: {device}")
        if usage_ratio > 0.8:
            print(f"\n⚠️  警告: GPU显存使用率已达 {usage_ratio*100:.1f}%")
            print("   如果遇到显存不足错误，建议重启Python进程以完全清理显存")
        return device
    
    # 多个GPU时，选择剩余可用显存最大的
    gpu_info = []
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        free_memory, total_memory = torch.cuda.mem_get_info(i)
        free_memory_gb = free_memory / (1024**3)
        total_memory_gb = total_memory / (1024**3)
        used_memory_gb = total_memory_gb - free_memory_gb
        usage_ratio = used_memory_gb / total_memory_gb if total_memory_gb > 0 else 0
        gpu_info.append({
            'id': i,
            'name': props.name,
            'total_memory': total_memory_gb,
            'used_memory': used_memory_gb,
            'free_memory': free_memory_gb,
            'usage_ratio': usage_ratio
        })
    
    # 按剩余显存大小排序，选择最大的
    best_gpu = max(gpu_info, key=lambda x: x['free_memory'])
    device = torch.device(f"cuda:{best_gpu['id']}")
    
    print(f"检测到 {num_gpus} 个GPU:")
    for gpu in gpu_info:
        marker = " <-- 已选择" if gpu['id'] == best_gpu['id'] else ""
        usage_warning = ""
        if gpu['usage_ratio'] > 0.9:
            usage_warning = " ⚠️ 显存使用率过高，建议重启Python进程清理显存"
        elif gpu['usage_ratio'] > 0.7:
            usage_warning = " ⚠️ 显存使用率较高"
        print(f"  GPU {gpu['id']}: {gpu['name']}")
        print(f"    总显存: {gpu['total_memory']:.2f} GB, 已用: {gpu['used_memory']:.2f} GB ({gpu['usage_ratio']*100:.1f}%), 剩余: {gpu['free_memory']:.2f} GB{marker}{usage_warning}")
    print(f"使用设备: {device} (剩余显存最大: {best_gpu['free_memory']:.2f} GB)")
    
    # 如果选择的GPU显存使用率过高，给出警告
    if best_gpu['usage_ratio'] > 0.8:
        print(f"\n⚠️  警告: 选择的GPU显存使用率已达 {best_gpu['usage_ratio']*100:.1f}%")
        print("   如果遇到显存不足错误，建议：")
        print("   1. 重启Python进程以完全清理显存")
        print("   2. 减小batch_size")
        print("   3. 使用其他显存使用率较低的GPU")
    
    return device

# 清理GPU缓存，解决显存碎片化问题
clear_gpu_cache()

device = select_best_gpu()
# 确保 data_load_operate 和 cls_visual 使用相同的 device
data_load_operate.set_device(device)
cls_visual.set_device(device)

cfg = EXPERIMENT_CONFIG
dataset_cfg = cfg["dataset"]
training_cfg = cfg["training"]
model_name = cfg["model_name"]
model_spec = MODEL_SPECS[model_name]

model_order = ['CPFormer', "CVSSN", "SQSFormer", "DBDA", "SSFTT", "SF", "S2FTNet", "HMSSF", "CWSST", "DynamicPSAM"]
if model_name not in model_order:
    model_order.append(model_name)
model_flag = model_order.index(model_name)
print(f"====================================== current model {model_name} =================================")

data_set_name = dataset_cfg.get("name", "HU18")
data_set_path = dataset_cfg.get("data_path", os.path.join(os.getcwd(), 'data'))

seed_list = training_cfg.get("seed_list", [0])

sampling_mode = dataset_cfg.get("sampling_mode", "ratio")
sampling_flag = 0 if sampling_mode == "ratio" else 1
use_validation = dataset_cfg.get("use_validation", False)
ratio_list = [
    dataset_cfg.get("train_ratio", 0.01),
    dataset_cfg.get("val_ratio", 0.001) if use_validation else 0.0
]
num_list = [
    dataset_cfg.get("train_num_per_class", 50),
    dataset_cfg.get("val_num_per_class", 0) if use_validation else 0
]
ratio = ratio_list[0] * 100 if sampling_flag == 0 else num_list[0]

patch_size = dataset_cfg.get("patch_size", 13)
patch_length = patch_size // 2
band_patch = dataset_cfg.get("band_patch", 3)
last_batch_flag = 0

# 重新组织保存路径结构：数据集目录 -> 模型目录 -> 分类图/结果txt/权重
# 基础路径：output/数据集名/模型名/
base_output_dir = os.path.join(os.getcwd(), 'output', data_set_name, model_name)
os.makedirs(base_output_dir, exist_ok=True)

# 子目录：分类图、结果txt
cls_map_dir = os.path.join(base_output_dir, 'cls_maps')
results_dir = os.path.join(base_output_dir, 'results')
os.makedirs(cls_map_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)


def calculate_model_complexity(net, model_spec, patch_channels, vector_channels, patch_size, band_patch=3, batch_size=1, model_name=None):
    """
    计算模型的参数量和FLOPs
    参数:
        net: 模型
        model_spec: 模型规格字典
        patch_channels: 补丁通道数
        vector_channels: 向量通道数（如果适用）
        patch_size: 补丁大小
        band_patch: 波段补丁大小（用于sf模式）
        batch_size: 批次大小（用于FLOPs计算）
        model_name: 模型名称（用于确定具体的输入shape）
    返回:
        params: 参数量（M）
        flops: FLOPs（M）
    """
    # 直接用9个if判断每个模型的输入格式
    # 注意：ptflops的input_shape不需要包含batch_size，它会自动处理
    # FlopCountAnalysis需要实际的tensor，会在后面创建
    if model_name == "CPFormer":
        # CPFormer: B P P C -> P P C
        input_shape = (patch_size, patch_size, patch_channels)
    elif model_name == "SQSFormer":
        # SQSFormer: B P P C -> P P C
        input_shape = (patch_size, patch_size, patch_channels)
    elif model_name == "DBDA":
        # DBDA: B 1 P P C -> 1 P P C
        input_shape = (1, patch_size, patch_size, patch_channels)
    elif model_name == "SSFTT":
        # SSFTT: B 1 P P C -> 1 P P C
        input_shape = (1, patch_size, patch_size, patch_channels)
    elif model_name == "HMSSF":
        # HMSSF: B 1 P P C -> 1 P P C
        input_shape = (1, patch_size, patch_size, patch_channels)
    elif model_name == "SF":
        # SF: B C P*P*near_band -> C P*P*near_band
        input_shape = (patch_channels, patch_size * patch_size * band_patch)
    elif model_name == "CVSSN":
        # CVSSN: B P P C 和 B C -> (P P C, C)
        input_shape = ((patch_size, patch_size, patch_channels), 
                      (vector_channels if vector_channels else patch_channels,))
    elif model_name == "S2FTNet":
        # S2FTNet: B 1 P P C 和 B C -> (1 P P C, C)
        input_shape = ((1, patch_size, patch_size, patch_channels), 
                      (vector_channels if vector_channels else patch_channels,))
    elif model_name == "CWSST":
        # CWSST: B P P C -> P P C
        input_shape = (patch_size, patch_size, patch_channels)
    elif model_name == "DynamicPSAM":
        # DynamicPSAMClassifier: B C H W（通道在前）
        input_shape = (patch_channels, patch_size, patch_size)
    elif model_name == "fix_csp_gqk_network_res_no_fusion":
        # fix_csp_gqk_network_res_no_fusion: B 1 P P C -> 1 P P C
        input_shape = (1, patch_size, patch_size, patch_channels)
    else:
        raise ValueError(f"未知的模型名称: {model_name}")
    
    # 判断是单输入还是多输入模型
    is_multi_input = isinstance(input_shape, tuple) and len(input_shape) == 2 and isinstance(input_shape[0], tuple)
    
    if is_multi_input:
        # 多输入模型：使用FlopCountAnalysis计算FLOPs，用sum计算参数量
        if not FVCORE_AVAILABLE:
            print(f"警告: 多输入模型({model_name})需要fvcore来计算FLOPs，但fvcore未安装")
            # 只计算参数量
            total_params = sum(p.numel() for p in net.parameters())
            params_m = total_params / (1000 ** 2)
            return params_m, 0.0
        
        try:
            net.eval()
            device = next(net.parameters()).device
            
            # 根据input_shape创建实际的tensor
            input_shape_1, input_shape_2 = input_shape
            # 创建dummy输入tensor（batch_size=1）
            inputs = (
                torch.randn(1, *input_shape_1).to(device),
                torch.randn(1, *input_shape_2).to(device)
            )
            
            # 使用FlopCountAnalysis计算FLOPs
            flops_counter = FlopCountAnalysis(net, inputs)
            flops_total = flops_counter.total()
            flops_m = flops_total / (1000 ** 2)  # 转换为M（百万）
            
            # 使用sum计算参数量
            total_params = sum(p.numel() for p in net.parameters())
            params_m = total_params / (1000 ** 2)
            
            return params_m, flops_m
        except Exception as e:
            print(f"多输入模型({model_name})FLOPs计算失败: {e}")
            # 至少计算参数量
            total_params = sum(p.numel() for p in net.parameters())
            params_m = total_params / (1000 ** 2)
            return params_m, 0.0
    else:
        # 单输入模型：使用get_model_complexity_info计算参数量和FLOPs
        try:
            flops, params = get_model_complexity_info(
                net,
                input_shape,
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False
            )
            
            # 转换为M（百万）
            params_m = params / (1000 ** 2)
            flops_m = flops / (1000 ** 2)
            
            return params_m, flops_m
        except Exception as e:
            # FLOPs计算失败，至少计算参数量
            print(f"单输入模型({model_name})FLOPs计算失败: {e}")
            print(f"  输入shape: {input_shape}")
            # 使用get_model_complexity_info失败时，用sum计算参数量作为备选
            total_params = sum(p.numel() for p in net.parameters())
            params_m = total_params / (1000 ** 2)
            return params_m, 0.0
    



def build_model(model_name, class_count, patch_channels, vector_channels, patch_size):
    dim = training_cfg.get("dim", 64)
    depth = training_cfg.get("depth", 2)
    heads = training_cfg.get("heads", 8)
    mlp_dim = training_cfg.get("mlp_dim", 128)
    dropout = training_cfg.get("dropout", 0.)
    num_groups = training_cfg.get("num_groups", 4)

    if model_name == "CPFormer":
        params = {
            "net": {
                "depth": depth,
                "heads": heads,
                "mlp_dim": mlp_dim,
                "kernal": 3,
                "padding": 1,
                "dropout": dropout,
                "dim": dim
            },
            "data": {
                "num_classes": class_count,
                "spectral_size": patch_channels
            }
        }
        return CPFormer.CPFormer(params)
    if model_name == "CVSSN":
        return CVSSN.CVSSN_(in_channels=patch_channels, h=patch_size, w=patch_size, num_classes=class_count)
    if model_name == "SQSFormer":
        params = {
            "net": {
                "depth": depth,
                "heads": heads,
                "mlp_dim": mlp_dim,
                "kernal": 3,
                "padding": 1,
                "dropout": dropout,
                "dim": dim
            },
            "data": {
                "num_classes": class_count,
                "patch_size": patch_size,
                "spectral_size": patch_channels
            }
        }
        return SQSFormer.SQSFormer(params)
    if model_name == "DBDA":
        return DBDA.DBDA_network_MISH(patch_channels, class_count)
    if model_name == "SSFTT":
        # in_channels=1, num_classes=NUM_CLASS, num_tokens=4, dim=64, depth=1, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1)
        return SSFTT.SSFTTnet(num_classes=class_count,dim=64, depth=2, heads=8, mlp_dim=128, dropout=0.1, emb_dropout=0.1)
    if model_name == "SF":
        # image_size, near_band, num_patches, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=1, dim_head = 16, dropout=0., emb_dropout=0., mode='ViT'):
        return SFModel.ViT(
            image_size=patch_size,
            near_band=band_patch,
            num_patches=patch_channels,
            num_classes=class_count,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=training_cfg.get("sf_dropout", 0.1),
            emb_dropout=training_cfg.get("sf_emb_dropout", 0.1),
            dim_head=training_cfg.get("sf_dim_head", 16),
            mode=training_cfg.get("sf_mode", "ViT")
        )
    if model_name == "S2FTNet":
        spectral_dim = vector_channels if vector_channels is not None else patch_channels# 200 
        return S2FTNet.S2FTNet(
            xy=patch_size,
            img_channels=spectral_dim,
            band=patch_channels, # 30
            num_classes=class_count,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout
        )
    if model_name == "HMSSF":
        return HMSSF3.LSFAT(num_classes=class_count, depth=depth)
    if model_name == "CWSST":
        return CWSST.CWSSTNet(
            in_channels=patch_channels,
            num_classes=class_count,
            embed_dim=dim,
            num_heads=heads,
            num_groups=num_groups,
            depth=depth,
            dropout_attn=dropout,
            dropout_ffn=dropout
        )
    
    
    raise ValueError(f"未支持的模型: {model_name}")


if __name__ == '__main__':

    data, gt = data_load_operate.load_data(data_set_name, data_set_path)
    data = data_load_operate.standardization(data)
    gt_reshape = gt.reshape(-1)
    height, width, _ = data.shape
    class_count = int(np.max(gt))

    data_sources = data_load_operate.prepare_data_sources(
        data=data,
        patch_length=patch_length,
        spec=model_spec,
        pca_components=dataset_cfg.get("pca_components")
    )
    patch_channels = data_sources.get("patch_channels", data.shape[-1])
    vector_channels = data_sources.get("vector_channels")

    batch_size = training_cfg.get("batch_size", 32)
    max_epoch = training_cfg.get("max_epoch", 150)
    learning_rate = training_cfg.get("learning_rate", 1e-3)
    loss = torch.nn.CrossEntropyLoss()
    
    def evaluate_loader(net, data_iter, input_mode):
        """
        在验证/测试集上评估模型，返回OA、平均loss、预测与标签
        """
        net.eval()
        total_loss = 0.0
        total_samples = 0
        correct = 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in data_iter:
                if input_mode == "spa":
                    X_spa, y = batch
                    X_spa, y = X_spa.to(device), y.to(device)
                    y_pred = net(X_spa)
                elif input_mode == "spa_spe":
                    X_spa, X_spe, y = batch
                    X_spa, X_spe, y = X_spa.to(device), X_spe.to(device), y.to(device)
                    y_pred = net(X_spa, X_spe)
                elif input_mode == "sf":
                    X_sf, y = batch
                    X_sf, y = X_sf.to(device), y.to(device)
                    y_pred = net(X_sf)
                else:
                    raise ValueError(f"未知的输入模式: {input_mode}")
                
                if isinstance(y_pred, (tuple, list)):
                    y_pred = y_pred[0]
                
                batch_loss = loss(y_pred, y.long())
                total_loss += batch_loss.item() * y.size(0)
                total_samples += y.size(0)
                
                preds = y_pred.argmax(dim=1)
                correct += (preds == y).sum().item()
                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())
        
        if total_samples == 0:
            return 0.0, 0.0, np.array([]), np.array([])
        avg_loss = total_loss / total_samples
        oa = correct / total_samples
        preds_np = torch.cat(all_preds).numpy()
        targets_np = torch.cat(all_targets).numpy()
        return oa, avg_loss, preds_np, targets_np
    
    # 获取模型参数（用于文件命名和结果保存）
    depth = training_cfg.get("depth", 2)
    dim = training_cfg.get("dim", 64)
    pca_components = dataset_cfg.get("pca_components", patch_channels)

    # 构建模型用于计算复杂度（不训练）
    temp_net = build_model(model_name, class_count, patch_channels, vector_channels, patch_size)
    params_m, flops_m = calculate_model_complexity(
        temp_net, model_spec, patch_channels, vector_channels, patch_size, 
        band_patch=band_patch, batch_size=1, model_name=model_name
    )
    print(f"模型参数量: {params_m:.3f} M, FLOPs: {flops_m:.3f} M")
    del temp_net
    torch.cuda.empty_cache()

    # 生成真实分类图（只生成一次）
    gt_map_path = os.path.join(cls_map_dir, f"{model_name}-{data_set_name}-{time_current}-gt")
    cls_visual.gt_cls_map(gt, gt_map_path)
    print(f"真实分类图已保存至: {gt_map_path}_gt.png")

    OA_ALL, AA_ALL, KPP_ALL = [], [], []
    VAL_OA_ALL = []
    EACH_ACC_ALL, Train_Time_ALL, Test_Time_ALL = [], [], []
    best_oa = 0.0
    best_seed = -1


    # 一次性打印数据划分与验证开关信息
    print(f"数据划分方式: {'ratio(按比例)' if sampling_flag == 0 else 'fixed(按数量)'} | 验证集: {'启用' if use_validation else '未启用'}")

    for curr_seed in seed_list:
        tic1 = time.perf_counter()
        
        # 采样训练/验证/测试集
        if use_validation:
            train_data_index, val_data_index, test_data_index, all_data_index = data_load_operate.sampling(
                ratio_list, num_list, gt_reshape, class_count, sampling_flag, use_validation=True
            )
            index = (train_data_index, val_data_index, test_data_index)
        else:
            train_data_index, test_data_index, all_data_index = data_load_operate.sampling(
                ratio_list, num_list, gt_reshape, class_count, sampling_flag, use_validation=False
            )
            index = (train_data_index, test_data_index)

        # 生成训练 / 验证 / 测试迭代器
        if use_validation:
            train_iter, val_iter, test_iter = data_load_operate.generate_iter_by_spec(
                data_sources,
                height,
                width,
                gt_reshape,
                index,
                patch_length,
                batch_size,
                model_spec,
                last_batch_flag,
                band_patch=band_patch
            )
            val_sample_num = len(val_iter.dataset)
        else:
            train_iter, test_iter = data_load_operate.generate_iter_by_spec(
                data_sources,
                height,
                width,
                gt_reshape,
                index,
                patch_length,
                batch_size,
                model_spec,
                last_batch_flag,
                band_patch=band_patch
            )
            val_sample_num = 0
        
        
        # 生成所有样本的迭代器（用于分类图可视化）
        total_iter = data_load_operate.generate_iter_total_by_spec(
            data_sources,
            height,
            width,
            gt_reshape,
            all_data_index,
            patch_length,
            batch_size,
            model_spec,
            band_patch=band_patch
        )

        net = build_model(model_name, class_count, patch_channels, vector_channels, patch_size).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

        best_state_dict = None
        best_val_oa = -1.0 if use_validation else None
        best_epoch = -1
        
        
        for epoch in range(max_epoch):
            train_acc_sum = trained_samples_counter = 0
            batch_counter ,train_loss_sum= 0, 0
            time_epoch = time.time()

            # 训练阶段
            net.train()
            if model_spec.get("input_mode") == "spa":
                for X_spa, y in train_iter:
                    X_spa, y = X_spa.to(device), y.to(device)
                    y_pred = net(X_spa)
                    if isinstance(y_pred, (tuple, list)):
                        y_pred = y_pred[0]
                    ls = loss(y_pred, y.long())
                    optimizer.zero_grad()
                    ls.backward()
                    optimizer.step()
                    train_loss_sum += ls.cpu().item()
                    train_acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
                    trained_samples_counter += y.shape[0]
                    batch_counter += 1
            elif model_spec.get("input_mode") == "spa_spe":
                for X_spa, X_spe, y in train_iter:
                    X_spa, X_spe, y = X_spa.to(device), X_spe.to(device), y.to(device)
                    y_pred = net(X_spa, X_spe)
                    if isinstance(y_pred, (tuple, list)):
                        y_pred = y_pred[0]
                    ls = loss(y_pred, y.long())
                    optimizer.zero_grad()
                    ls.backward()
                    optimizer.step()

                    train_loss_sum += ls.cpu().item()
                    train_acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
                    trained_samples_counter += y.shape[0]
                    batch_counter += 1
            elif model_spec.get("input_mode") == "sf":
                for X_sf, y in train_iter:
                    X_sf, y = X_sf.to(device), y.to(device)
                    y_pred = net(X_sf)
                    ls = loss(y_pred, y.long())
                    optimizer.zero_grad()
                    ls.backward()
                    optimizer.step()
                    train_loss_sum += ls.cpu().item()
                    train_acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
                    trained_samples_counter += y.shape[0]
                    batch_counter += 1

            torch.cuda.empty_cache()
            
            print('epoch: %d, training_sampler_num: %d, val_sampler_num: %d, batch_count: %.2f, train loss: %.6f, tarin loss sum: %.6f, '\
                    'train acc: %.3f, train_acc_sum: %.1f, time: %.1f sec' %\
                    (epoch + 1, trained_samples_counter, val_sample_num, batch_counter, train_loss_sum / batch_counter, train_loss_sum,\
                    train_acc_sum / trained_samples_counter, train_acc_sum, time.time() - time_epoch))

            # 验证阶段（如启用）
            if use_validation:
                val_oa, val_loss, _, _ = evaluate_loader(net, val_iter, model_spec.get("input_mode"))
                if val_oa > best_val_oa:
                    best_val_oa = val_oa
                    best_state_dict = net.state_dict()
                    best_epoch = epoch + 1
                # print(f'  验证集: OA={val_oa:.4f}, loss={val_loss:.4f} (best OA={best_val_oa:.4f} @epoch {best_epoch})')

        # 若未使用验证集，则默认使用最后一轮权重
        if not use_validation:
            best_state_dict = net.state_dict()
            best_epoch = max_epoch
        
        toc1 = time.perf_counter()
        training_time = toc1 - tic1
        Train_Time_ALL.append(training_time)

        # 测试前加载最优验证权重（如有）
        if best_state_dict is not None:
            net.load_state_dict(best_state_dict)

        print("\n使用最优模型权重进行测试")

        # 测试阶段（使用验证集最优权重）
        tic_test = time.perf_counter()
        test_oa_dummy, test_loss_dummy, pred_test, y_gt = evaluate_loader(net, test_iter, model_spec.get("input_mode"))
        testing_time = time.perf_counter() - tic_test

        y_gt = y_gt.astype(int)
        OA = metrics.accuracy_score(y_gt, pred_test)
       
        confusion_matrix = metrics.confusion_matrix(y_gt, pred_test)
        print("confusion_matrix\n{}".format(confusion_matrix))
        ECA, AA = evaluation.AA_ECA(confusion_matrix)
        print("--------------------------------ECA\n{}".format(ECA))
        kappa = metrics.cohen_kappa_score(pred_test, y_gt)
        cls_report = evaluation.claification_report(y_gt, pred_test, data_set_name)
        print("classification_report\n{}".format(cls_report))


        # 生成预测分类图
        # 将input_mode转换为cls_visual需要的model_type_flag: spa->1, spa_spe->3, sf->2
        input_mode = model_spec.get("input_mode")
        if input_mode == "spa":
            model_type_flag = 1
        elif input_mode == "spa_spe":
            model_type_flag = 3
        elif input_mode == "sf":
            model_type_flag = 2
        else:
            model_type_flag = 1
        
        # 生成预测分类图文件名：模型名-数据名-时间-随机种子-样本数-patchsize-pca降维后通道数-编码器数-嵌入维度-batch批次-学习率-epoch
        pred_map_filename = f"{model_name}-{data_set_name}-{time_current}-seed{curr_seed}-s{num_list[0]}-ps{patch_size}-pca{pca_components}-enc{depth}-dim{dim}-bs{batch_size}-lr{learning_rate}-ep{max_epoch}"
        pred_map_path = os.path.join(cls_map_dir, pred_map_filename)
        
        sample_list2 = [total_iter, all_data_index]
        cls_visual.pred_cls_map_dl(sample_list2, net, gt, pred_map_path, model_type_flag)
        print(f"预测分类图已保存至: {pred_map_path}_2_pre.png")
        
        # 生成分类图后立即清理内存
        # 删除total_iter和sample_list2以释放内存
        del total_iter, sample_list2
        torch.cuda.empty_cache()
        gc.collect()
        
        Test_Time_ALL.append(testing_time)
        print(f"---------------- OA={OA}, AA={AA}, kappa={kappa} ----------------")
        if use_validation:
            print(f"=============== 验证集最佳OA: {best_val_oa:.4f} @epoch {best_epoch}=================")

        # 判断是否需要更新最优权重
        is_best = OA > best_oa
        if is_best:
            prev_best_oa = best_oa
            prev_best_seed = best_seed
            best_oa = OA
            best_seed = curr_seed
            print(f'  测试集性能提升: 新OA={OA:.4f} (之前最佳={prev_best_oa:.4f}于seed {prev_best_seed})')
        
        # 清理内存，为后续操作腾出空间
        torch.cuda.empty_cache()
        gc.collect()

        # 生成结果文件名
        results_filename = f"{model_name}-{data_set_name}-{time_current}-seed{curr_seed}-s{num_list[0]}-ps{patch_size}-pca{pca_components}-enc{depth}-dim{dim}-bs{batch_size}-lr{learning_rate}-ep{max_epoch}.txt"
        results_file_path = os.path.join(results_dir, results_filename)
        
        with open(results_file_path, 'w') as f:
            # 写入所有参数和结果
            f.write("=" * 80 + "\n")
            f.write("模型配置和训练参数\n")
            f.write("=" * 80 + "\n")
            f.write(f"模型名称: {model_name}\n")
            f.write(f"数据集名称: {data_set_name}\n")
            f.write(f"时间戳: {time_current}\n")
            f.write(f"随机种子: {curr_seed}\n")
            f.write(f"\n数据集参数:\n")
            f.write(f"  每类训练样本数: {num_list[0]}\n")
            f.write(f"  补丁大小: {patch_size}\n")
            f.write(f"  PCA降维后通道数: {pca_components}\n")
            f.write(f"  原始通道数: {data.shape[-1]}\n")
            f.write(f"  补丁通道数: {patch_channels}\n")
            if vector_channels:
                f.write(f"  向量通道数: {vector_channels}\n")
            f.write(f"  类别数: {class_count}\n")
            f.write(f"\n模型参数:\n")
            f.write(f"  编码器层数: {depth}\n")
            f.write(f"  注意力头数: {training_cfg.get('heads', 8)}\n")
            f.write(f"  嵌入维度: {dim}\n")
            f.write(f"  MLP维度: {training_cfg.get('mlp_dim', 128)}\n")
            f.write(f"  Dropout: {training_cfg.get('dropout', 0.)}\n")
            if model_name == "SF":
                f.write(f"  SF模式: {training_cfg.get('sf_mode', 'ViT')}\n")
                f.write(f"  SF Dropout: {training_cfg.get('sf_dropout', 0.1)}\n")
            if model_name == "CWSST":
                f.write(f"  组数: {training_cfg.get('num_groups', 4)}\n")
            f.write(f"\n训练参数:\n")
            f.write(f"  批次大小: {batch_size}\n")
            f.write(f"  学习率: {learning_rate}\n")
            f.write(f"  训练轮数: {max_epoch}\n")
            f.write(f"  采样模式: {sampling_mode}\n")
            f.write(f"  是否使用验证集: {use_validation}\n")
            f.write(f"\n模型复杂度:\n")
            f.write(f"  参数量: {params_m:.3f} M\n")
            f.write(f"  FLOPs: {flops_m:.3f} M\n")
            f.write("\n" + "=" * 80 + "\n")
            f.write("训练结果\n")
            f.write("=" * 80 + "\n")
            f.write(f"训练时间: {training_time:.2f} 秒\n")
            f.write(f"测试时间: {testing_time:.5f} 秒\n")
            if use_validation:
                f.write(f"验证集最佳OA: {best_val_oa:.4f} @epoch {best_epoch}\n")
            f.write(f"\n性能指标:\n")
            f.write(f"OA (Overall Accuracy): {OA:.4f}\n")
            f.write(f"AA (Average Accuracy): {AA:.4f}\n")
            f.write(f"Kappa: {kappa:.4f}\n")
            f.write(f"每类准确率: {ECA}\n")
            f.write("\n混淆矩阵:\n")
            f.write('{}'.format(confusion_matrix))
            f.write('\n\n分类报告:\n')
            f.write('{}'.format(cls_report))

        OA_ALL.append(OA)
        AA_ALL.append(AA)
        KPP_ALL.append(kappa)
        EACH_ACC_ALL.append(ECA)
        if use_validation:
            VAL_OA_ALL.append(best_val_oa)
        
        torch.cuda.empty_cache()
        if use_validation:
            del val_iter
        del net, train_iter, test_iter
    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    if use_validation:
        VAL_OA_ALL = np.array(VAL_OA_ALL)
    EACH_ACC_ALL = np.array(EACH_ACC_ALL)
    Train_Time_ALL = np.array(Train_Time_ALL)
    Test_Time_ALL = np.array(Test_Time_ALL)

    np.set_printoptions(precision=4)
    print("\n====================Mean result of {} times runs =========================".format(len(seed_list)))
    print('List of OA:', list(OA_ALL))
    print('List of AA:', list(AA_ALL))
    print('List of KPP:', list(KPP_ALL))
    if use_validation:
        print('List of Val OA:', list(VAL_OA_ALL))
    print('OA=', round(np.mean(OA_ALL) * 100, 2), '+-', round(np.std(OA_ALL) * 100, 2))
    print('AA=', round(np.mean(AA_ALL) * 100, 2), '+-', round(np.std(AA_ALL) * 100, 2))
    print('Kpp=', round(np.mean(KPP_ALL) * 100, 2), '+-', round(np.std(KPP_ALL) * 100, 2))
    if use_validation:
        print('Val OA=', round(np.mean(VAL_OA_ALL) * 100, 2), '+-', round(np.std(VAL_OA_ALL) * 100, 2))
    print('Acc per class=', np.mean(EACH_ACC_ALL, 0), '+-', np.std(EACH_ACC_ALL, 0))

    print("Average training time=", round(np.mean(Train_Time_ALL), 2), '+-', round(np.std(Train_Time_ALL), 3))
    print("Average testing time=", round(np.mean(Test_Time_ALL), 5), '+-', round(np.std(Test_Time_ALL), 5))

    # 保存汇总结果
    summary_filename = f"{model_name}-{data_set_name}-{time_current}-summary-s{num_list[0]}-ps{patch_size}-pca{pca_components}-enc{depth}-dim{dim}-bs{batch_size}-lr{learning_rate}-ep{max_epoch}.txt"
    summary_file_path = os.path.join(results_dir, summary_filename)
    
    with open(summary_file_path, 'w') as f:
        # 写入所有参数
        f.write("=" * 80 + "\n")
        f.write("模型配置和训练参数（汇总）\n")
        f.write("=" * 80 + "\n")
        f.write(f"模型名称: {model_name}\n")
        f.write(f"数据集名称: {data_set_name}\n")
        f.write(f"时间戳: {time_current}\n")
        f.write(f"随机种子列表: {seed_list}\n")
        f.write(f"\n数据集参数:\n")
        f.write(f"  每类训练样本数: {num_list[0]}\n")
        f.write(f"  补丁大小: {patch_size}\n")
        f.write(f"  PCA降维后通道数: {pca_components}\n")
        f.write(f"  原始通道数: {data.shape[-1]}\n")
        f.write(f"  补丁通道数: {patch_channels}\n")
        if vector_channels:
            f.write(f"  向量通道数: {vector_channels}\n")
        f.write(f"  类别数: {class_count}\n")
        f.write(f"\n模型参数:\n")
        f.write(f"  编码器层数: {depth}\n")
        f.write(f"  注意力头数: {training_cfg.get('heads', 8)}\n")
        f.write(f"  嵌入维度: {dim}\n")
        f.write(f"  MLP维度: {training_cfg.get('mlp_dim', 128)}\n")
        f.write(f"  Dropout: {training_cfg.get('dropout', 0.)}\n")
        if model_name == "SF":
            f.write(f"  SF模式: {training_cfg.get('sf_mode', 'ViT')}\n")
            f.write(f"  SF Dropout: {training_cfg.get('sf_dropout', 0.1)}\n")
        if model_name == "CWSST":
            f.write(f"  组数: {training_cfg.get('num_groups', 4)}\n")
        f.write(f"\n训练参数:\n")
        f.write(f"  批次大小: {batch_size}\n")
        f.write(f"  学习率: {learning_rate}\n")
        f.write(f"  训练轮数: {max_epoch}\n")
        f.write(f"  采样模式: {sampling_mode}\n")
        f.write(f"  是否使用验证集: {use_validation}\n")
        f.write(f"\n模型复杂度:\n")
        f.write(f"  参数量: {params_m:.3f} M\n")
        f.write(f"  FLOPs: {flops_m:.3f} M\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"多次运行结果汇总（{len(seed_list)}次）\n")
        f.write("=" * 80 + "\n")
        f.write(f'OA列表: {list(OA_ALL)}\n')
        f.write(f'AA列表: {list(AA_ALL)}\n')
        f.write(f'Kappa列表: {list(KPP_ALL)}\n')
        if use_validation:
            f.write(f'Val OA列表: {list(VAL_OA_ALL)}\n')
        f.write(f'OA均值±标准差: {round(np.mean(OA_ALL) * 100, 2)}±{round(np.std(OA_ALL) * 100, 2)}%\n')
        f.write(f'AA均值±标准差: {round(np.mean(AA_ALL) * 100, 2)}±{round(np.std(AA_ALL) * 100, 2)}%\n')
        f.write(f'Kappa均值±标准差: {round(np.mean(KPP_ALL) * 100, 2)}±{round(np.std(KPP_ALL) * 100, 2)}%\n')
        if use_validation:
            f.write(f'Val OA均值±标准差: {round(np.mean(VAL_OA_ALL) * 100, 2)}±{round(np.std(VAL_OA_ALL) * 100, 2)}%\n')
        f.write(f'每类准确率均值±标准差:\n{np.mean(EACH_ACC_ALL, 0)}±{np.std(EACH_ACC_ALL, 0)}\n')
        f.write(f"\n平均训练时间: {round(np.mean(Train_Time_ALL), 2)}±{round(np.std(Train_Time_ALL), 3)} 秒\n")
        f.write(f"平均测试时间: {round(np.mean(Test_Time_ALL), 5)}±{round(np.std(Test_Time_ALL), 5)} 秒\n")
        f.write(f"\n最优结果: seed={best_seed}, OA={best_oa:.4f}\n")
