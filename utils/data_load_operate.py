# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2023-Apr
# @Address  : Time Lab @ SDU
# @FileName : data_load_operate.py
# @Project  : AMS-M2ESL (HSIC), IEEE TGRS
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import os
import math
import torch
import numpy as np
import spectral as spy
import scipy.io as sio
import torch.utils.data as Data
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
import h5py  # 用于读取 Matlab v7.3 (HDF5) 格式的 .mat 文件


# 设备选择逻辑：自动选择剩余可用显存最大的GPU
def select_best_gpu():
    """
    自动选择剩余可用显存最大的GPU
    返回: torch.device 对象
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return torch.device("cpu")
    
    if num_gpus == 1:
        return torch.device("cuda:0")
    
    # 多个GPU时，选择剩余可用显存最大的
    gpu_info = []
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        free_memory, total_memory = torch.cuda.mem_get_info(i)
        free_memory_gb = free_memory / (1024**3)
        gpu_info.append({
            'id': i,
            'name': props.name,
            'free_memory': free_memory_gb
        })
    
    # 按剩余显存大小排序，选择最大的
    best_gpu = max(gpu_info, key=lambda x: x['free_memory'])
    return torch.device(f"cuda:{best_gpu['id']}")

# 延迟device选择：不在模块导入时选择，而是在实际使用时选择
# 这样可以避免与train_.py中的device选择冲突
_device = None

def get_device():
    """
    获取当前device，如果未设置则自动选择剩余显存最大的GPU
    返回: torch.device 对象
    """
    global _device
    if _device is None:
        _device = select_best_gpu()
    return _device

def set_device(device):
    """
    设置device（由train_.py调用，确保使用相同的device）
    参数: device - torch.device 对象
    """
    global _device
    _device = device

# 为了向后兼容，提供一个device变量，但实际使用时通过get_device()获取
# 注意：所有使用device的地方都需要改为get_device()


def load_data(data_set_name, data_path):
    # def load_data(data_set_name, data_path):
    if data_set_name == 'IP':
        data = sio.loadmat(os.path.join(data_path,  'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path , 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif data_set_name == 'UP':
        data = sio.loadmat(os.path.join(data_path,  'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path,  'PaviaU_gt.mat'))['paviaU_gt']
    elif data_set_name == "SA":
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    if data_set_name == "BOT":
        data = sio.loadmat(os.path.join(data_path, 'Botswana.mat'))['Botswana']
        labels = sio.loadmat(os.path.join(data_path, 'Botswana_gt.mat'))['Botswana_gt']
    if data_set_name == "KSC":
        data = sio.loadmat(os.path.join(data_path, 'KSC.mat'))['KSC']
        labels = sio.loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt']
    if data_set_name == "CH":
        # 原始代码：
        # data = h5py.File(os.path.join(data_path, 'Chikusei.mat'))['chikusei'][:].transpose(1, 2, 0)
        # labels = sio.loadmat(os.path.join(data_path, 'Chikusei_gt.mat'))['GT'][0][0][0]
        #
        # 这里保持原有逻辑，仅补充 h5py 导入（已在文件头部添加），避免 v7.3 读取问题。
        with h5py.File(os.path.join(data_path, 'Chikusei.mat'), "r") as f_ch:
            data = f_ch['chikusei'][:].transpose(1, 2, 0)
        labels = sio.loadmat(os.path.join(data_path, 'Chikusei_gt.mat'))['GT'][0][0][0]
    if data_set_name == "HU":
        data = sio.loadmat(os.path.join(data_path,"DFC2013_Houston.mat"))['Houston']  # (349, 1905, 144)
        labels = sio.loadmat(os.path.join(data_path,"DFC2013_Houston_gt.mat"))['Houston_gt']  # (349, 1905)
    if data_set_name == "HU18":
        # 原始加载方式（Matlab v7.3 的 .mat 会触发 NotImplementedError）：
        # data = sio.loadmat(os.path.join(data_path, 'HoustonU.mat'))['houstonU']
        # labels = sio.loadmat(os.path.join(data_path, 'HoustonU_gt.mat'))['houstonU_gt']
        #
        # 新增：优先使用 scipy.io.loadmat（兼容 v7），如果遇到 v7.3 报错，则自动切换到 h5py 读取。
        data_mat_path = os.path.join(data_path, 'HoustonU.mat')
        label_mat_path = os.path.join(data_path, 'HoustonU_gt.mat')

        # 读取数据立方体
        try:
            data = sio.loadmat(data_mat_path)['houstonU']
        except NotImplementedError:
            # v7.3 情况，使用 h5py 读取 HDF5 结构
            with h5py.File(data_mat_path, "r") as f_hu18:
                # 常见命名为 'houstonU'，否则退而求其次取第一个数据集
                if 'houstonU' in f_hu18.keys():
                    data = f_hu18['houstonU'][:]
                elif 'HoustonU' in f_hu18.keys():
                    data = f_hu18['HoustonU'][:]
                else:
                    first_key = list(f_hu18.keys())[0]
                    data = f_hu18[first_key][:]

        # 读取标签
        try:
            labels = sio.loadmat(label_mat_path)['houstonU_gt']
        except NotImplementedError:
            with h5py.File(label_mat_path, "r") as f_hu18_gt:
                if 'houstonU_gt' in f_hu18_gt.keys():
                    labels = f_hu18_gt['houstonU_gt'][:]
                elif 'HoustonU_gt' in f_hu18_gt.keys():
                    labels = f_hu18_gt['HoustonU_gt'][:]
                else:
                    first_key = list(f_hu18_gt.keys())[0]
                    labels = f_hu18_gt[first_key][:]

        # 统一 HU18 数据、标签的形状：
        # - 数据：统一为 (H, W, C)，与 IP/UP/HU 等保持一致，避免空间维度/通道维度错位
        # - 标签：去掉多余维度，统一为 (H, W)
        data = np.array(data)
        if data.ndim == 3:
            # 若第一维明显是“通道数”（通常远小于 H、W），认为当前是 (C, H, W)，需要转为 (H, W, C)
            if data.shape[0] < data.shape[1] and data.shape[0] < data.shape[2]:
                data = np.transpose(data, (1, 2, 0))

        labels = np.array(labels)
        if labels.ndim > 2:
            labels = np.squeeze(labels)

    return data, labels


def load_HU_data(data_path):
    data = sio.loadmat(os.path.join(data_path, 'HU13_tif', "Houston13_data.mat"))['Houston13_data']
    labels_train = sio.loadmat(os.path.join(data_path, 'HU13_tif', "Houston13_gt_train.mat"))['Houston13_gt_train']
    labels_test = sio.loadmat(os.path.join(data_path, 'HU13_tif', "Houston13_gt_test.mat"))['Houston13_gt_test']

    return data, labels_train, labels_test


def standardization(data):
    height, width, bands = data.shape
    data = np.reshape(data, [height * width, bands])
    # data=preprocessing.scale(data) #
    # data = preprocessing.MinMaxScaler().fit_transform(data)
    data = preprocessing.StandardScaler().fit_transform(data)  #

    data = np.reshape(data, [height, width, bands])
    return data


def sampling(ratio_list, num_list, gt_reshape, class_count, Flag, use_validation=False): # 1
    all_label_index_dict, train_label_index_dict, val_label_index_dict, test_label_index_dict = {}, {}, {}, {}
    all_label_index_list, train_label_index_list, val_label_index_list, test_label_index_list = [], [], [], []

    for cls in range(class_count):  # [0-15]
        cls_index = np.where(gt_reshape == cls + 1)[0]
        all_label_index_dict[cls] = list(cls_index)

        np.random.shuffle(cls_index)

        if Flag == 0:  # Fixed proportion for each category
            if use_validation:
                # 划分训练集、验证集，测试集为剩余
                total_len = len(cls_index)
                train_index_flag = max(int(ratio_list[0] * total_len), 3)  # at least 3 samples per class
                val_index_flag = train_index_flag + max(int(ratio_list[1] * total_len), 2)  # at least 2 samples per class
                test_index_flag = total_len  # 剩余即测试集
            else:
                # 只划分训练集和测试集
                train_index_flag = max(int(ratio_list[0] * len(cls_index)), 3)  # at least 3 samples per class
                val_index_flag = train_index_flag
                test_index_flag = len(cls_index)
        # Split by num per class
        elif Flag == 1:  # Fixed quantity per category
            if use_validation:
                # 划分训练集、验证集，测试集为剩余
                train_index_flag = min(num_list[0], len(cls_index)) if num_list[0] > 0 else 0
                val_index_flag = min(train_index_flag + num_list[1], len(cls_index)) if len(num_list) > 1 and num_list[1] > 0 else train_index_flag
                test_index_flag = len(cls_index)  # 剩余即测试集
            else:
                # 只划分训练集和测试集
                train_index_flag = min(num_list[0], len(cls_index)) if num_list[0] > 0 else 0
                val_index_flag = train_index_flag
                test_index_flag = len(cls_index)

        train_label_index_dict[cls] = list(cls_index[:train_index_flag])
        val_label_index_dict[cls] = list(cls_index[train_index_flag:val_index_flag]) if use_validation else []
        test_label_index_dict[cls] = list(cls_index[val_index_flag:test_index_flag])

        train_label_index_list += train_label_index_dict[cls]
        val_label_index_list += val_label_index_dict[cls]
        test_label_index_list += test_label_index_dict[cls]
        all_label_index_list += all_label_index_dict[cls]

    if use_validation:
        return train_label_index_list, val_label_index_list, test_label_index_list, all_label_index_list
    else:
        return train_label_index_list, test_label_index_list, all_label_index_list


def sampling_disjoint(gt_train_re, gt_test_re, class_count):
    all_label_index_dict, train_label_index_dict, test_label_index_dict = {}, {}, {}
    all_label_index_list, train_label_index_list, test_label_index_list = [], [], []

    for cls in range(class_count):
        cls_index_train = np.where(gt_train_re == cls + 1)[0]
        cls_index_test = np.where(gt_test_re == cls + 1)[0]

        train_label_index_dict[cls] = list(cls_index_train)
        test_label_index_dict[cls] = list(cls_index_test)

        train_label_index_list += train_label_index_dict[cls]
        test_label_index_list += test_label_index_dict[cls]
        all_label_index_list += (train_label_index_dict[cls] + test_label_index_dict[cls])

    return train_label_index_list, test_label_index_list, all_label_index_list


def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


def pad_with_reflect(data, pad_length):
    if pad_length <= 0:
        return data.astype(np.float32)
    return np.pad(
        data.astype(np.float32),
        ((pad_length, pad_length), (pad_length, pad_length), (0, 0)),
        mode='reflect'
    )


def prepare_data_sources(data, patch_length, spec, pca_components=None, pad_mode='reflect'):
    data = data.astype(np.float32)
    patch_source = data
    vector_source = data
    pca_data = None
    needs_patch_pca = spec.get("use_pca_patch", False)
    needs_vector_pca = spec.get("vector_use_pca", False)

    if needs_patch_pca or needs_vector_pca:
        if pca_components is None:
            raise ValueError("当前模型需要PCA降维，请在配置文件中设置 pca_components。")
        target_dim = max(1, min(pca_components, data.shape[-1]))
        if target_dim != pca_components:
            print(f"[PCA] 目标维度 {pca_components} 超过原始光谱维度 {data.shape[-1]}，自动调整为 {target_dim}")
        pca_data = applyPCA(data, target_dim).astype(np.float32)
    if needs_patch_pca:
        patch_source = pca_data
    if spec.get("input_mode") == "spa_spe":
        vector_source = pca_data if needs_vector_pca else data

    sources = {
        "patch_data_padded": pad_with_reflect(patch_source, patch_length),
        "patch_channels": int(patch_source.shape[-1]),
        "raw_data": data,
    }
    if spec.get("input_mode") == "spa_spe":
        sources["vector_data_padded"] = pad_with_reflect(vector_source, patch_length)
        sources["vector_channels"] = int(vector_source.shape[-1])
    if spec.get("input_mode") == "sf":
        sources["sf_channels"] = int(patch_source.shape[-1])
    return sources


def _build_sf_tensor(patches, band_patch):
    if band_patch % 2 == 0:
        raise ValueError("band_patch 必须为奇数。")
    b, h, w, c = patches.shape
    flattened = patches.view(b, h * w, c)
    radius = band_patch // 2
    windows = []
    for shift in range(-radius, radius + 1):
        windows.append(torch.roll(flattened, shifts=shift, dims=2))
    stacked = torch.cat(windows, dim=1)  # B, (P*P*band_patch), C
    return stacked.permute(0, 2, 1)  # B, C, P*P*band_patch


def generate_iter_by_spec(data_sources, hsi_h, hsi_w, label_reshape, index, patch_length,
                          batch_size, spec, last_batch_flag, band_patch=None):
    if spec.get("input_mode") == "sf":
        band_patch = band_patch or 3

    patch_tensor = torch.from_numpy(data_sources["patch_data_padded"]).float().to(get_device())
    vector_tensor = None
    if spec.get("input_mode") == "spa_spe":
        vector_tensor = torch.from_numpy(data_sources["vector_data_padded"]).float().to(get_device())

    # 确定是否使用验证集
    use_validation = len(index) > 2
    
    # 提取标签
    train_labels = label_reshape[index[0]] - 1
    y_tensor_train = torch.from_numpy(train_labels).float()
    
    if use_validation:
        # 有验证集的情况
        val_labels = label_reshape[index[1]] - 1
        test_labels = label_reshape[index[2]] - 1
        y_tensor_val = torch.from_numpy(val_labels).float()
        y_tensor_test = torch.from_numpy(test_labels).float()
    else:
        # 没有验证集的情况
        test_labels = label_reshape[index[1]] - 1
        y_tensor_test = torch.from_numpy(test_labels).float()

    # 创建数据集
    if spec.get("input_mode") == "spa":
        train_samples = HSI_create_pathes(patch_tensor, hsi_h, hsi_w, index[0], patch_length, 1)
        
        if use_validation:
            val_samples = HSI_create_pathes(patch_tensor, hsi_h, hsi_w, index[1], patch_length, 1)
            test_samples = HSI_create_pathes(patch_tensor, hsi_h, hsi_w, index[2], patch_length, 1)
        else:
            test_samples = HSI_create_pathes(patch_tensor, hsi_h, hsi_w, index[1], patch_length, 1)
            
        if spec.get("needs_3d"):
            train_samples = train_samples.unsqueeze(1)
            if use_validation:
                val_samples = val_samples.unsqueeze(1)
            test_samples = test_samples.unsqueeze(1)
            
        torch_dataset_train = Data.TensorDataset(train_samples, y_tensor_train)
        if use_validation:
            torch_dataset_val = Data.TensorDataset(val_samples, y_tensor_val)
        torch_dataset_test = Data.TensorDataset(test_samples, y_tensor_test)
        
    elif spec.get("input_mode") == "spa_spe":
        train_spa = HSI_create_pathes(patch_tensor, hsi_h, hsi_w, index[0], patch_length, 1)
        train_spe = HSI_create_pathes(vector_tensor, hsi_h, hsi_w, index[0], patch_length, 2)
        
        if use_validation:
            val_spa = HSI_create_pathes(patch_tensor, hsi_h, hsi_w, index[1], patch_length, 1)
            val_spe = HSI_create_pathes(vector_tensor, hsi_h, hsi_w, index[1], patch_length, 2)
            test_spa = HSI_create_pathes(patch_tensor, hsi_h, hsi_w, index[2], patch_length, 1)
            test_spe = HSI_create_pathes(vector_tensor, hsi_h, hsi_w, index[2], patch_length, 2)
        else:
            test_spa = HSI_create_pathes(patch_tensor, hsi_h, hsi_w, index[1], patch_length, 1)
            test_spe = HSI_create_pathes(vector_tensor, hsi_h, hsi_w, index[1], patch_length, 2)
            
        if spec.get("needs_3d"):
            train_spa = train_spa.unsqueeze(1) # torch.Size([160, 1, 9, 9, 120])
            if use_validation:
                val_spa = val_spa.unsqueeze(1)
            test_spa = test_spa.unsqueeze(1)
            
        torch_dataset_train = Data.TensorDataset(train_spa, train_spe, y_tensor_train)
        if use_validation:
            torch_dataset_val = Data.TensorDataset(val_spa, val_spe, y_tensor_val)
        torch_dataset_test = Data.TensorDataset(test_spa, test_spe, y_tensor_test)
        
    elif spec.get("input_mode") == "sf":
        train_patches = HSI_create_pathes(patch_tensor, hsi_h, hsi_w, index[0], patch_length, 1)
        train_sf = _build_sf_tensor(train_patches, band_patch)
        
        if use_validation:
            val_patches = HSI_create_pathes(patch_tensor, hsi_h, hsi_w, index[1], patch_length, 1)
            val_sf = _build_sf_tensor(val_patches, band_patch)
            test_patches = HSI_create_pathes(patch_tensor, hsi_h, hsi_w, index[2], patch_length, 1)
            test_sf = _build_sf_tensor(test_patches, band_patch)
        else:
            test_patches = HSI_create_pathes(patch_tensor, hsi_h, hsi_w, index[1], patch_length, 1)
            test_sf = _build_sf_tensor(test_patches, band_patch)
            
        torch_dataset_train = Data.TensorDataset(train_sf, y_tensor_train)
        if use_validation:
            torch_dataset_val = Data.TensorDataset(val_sf, y_tensor_val)
        torch_dataset_test = Data.TensorDataset(test_sf, y_tensor_test)
        
    else:
        raise ValueError(f"未知的输入模式: {spec.get('input_mode')}")

    drop_last = last_batch_flag == 1
    train_iter = Data.DataLoader(torch_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0,
                                 drop_last=drop_last)
    
    if use_validation:
        val_iter = Data.DataLoader(torch_dataset_val, batch_size=batch_size, shuffle=False, num_workers=0,
                                  drop_last=drop_last)
        test_iter = Data.DataLoader(torch_dataset_test, batch_size=batch_size, shuffle=False, num_workers=0,
                                    drop_last=drop_last)
        return train_iter, val_iter, test_iter
    else:
        test_iter = Data.DataLoader(torch_dataset_test, batch_size=batch_size, shuffle=False, num_workers=0,
                                    drop_last=drop_last)
        return train_iter, test_iter


def generate_iter_total_by_spec(data_sources, hsi_h, hsi_w, label_reshape, index, patch_length,
                                batch_size, spec, band_patch=None):
    """
    生成所有样本的迭代器（用于可视化分类图）
    参数:
        data_sources: 数据源字典
        hsi_h, hsi_w: 高光谱图像的高度和宽度
        label_reshape: 标签的一维数组
        index: 所有样本的索引（all_data_index）
        patch_length: 补丁长度
        batch_size: 批次大小
        spec: 模型规格字典
        band_patch: 波段补丁大小（用于sf模式）
    返回:
        total_iter: 所有样本的数据加载器
    """
    if spec.get("input_mode") == "sf":
        band_patch = band_patch or 3

    patch_tensor = torch.from_numpy(data_sources["patch_data_padded"]).float().to(get_device())
    vector_tensor = None
    if spec.get("input_mode") == "spa_spe":
        vector_tensor = torch.from_numpy(data_sources["vector_data_padded"]).float().to(get_device())

    if len(index) < label_reshape.shape[0]:
        total_labels = label_reshape[index] - 1
    else:
        total_labels = np.zeros(label_reshape.shape)
    y_tensor_total = torch.from_numpy(total_labels).float()

    if spec.get("input_mode") == "spa":
        total_samples = HSI_create_pathes(patch_tensor, hsi_h, hsi_w, index, patch_length, 1)
        if spec.get("needs_3d"):
            total_samples = total_samples.unsqueeze(1)
        torch_dataset_total = Data.TensorDataset(total_samples, y_tensor_total)
    elif spec.get("input_mode") == "spa_spe":
        total_spa = HSI_create_pathes(patch_tensor, hsi_h, hsi_w, index, patch_length, 1)
        total_spe = HSI_create_pathes(vector_tensor, hsi_h, hsi_w, index, patch_length, 2)
        if spec.get("needs_3d"):
            total_spa = total_spa.unsqueeze(1)
        torch_dataset_total = Data.TensorDataset(total_spa, total_spe, y_tensor_total)
    elif spec.get("input_mode") == "sf":
        total_patches = HSI_create_pathes(patch_tensor, hsi_h, hsi_w, index, patch_length, 1)
        total_sf = _build_sf_tensor(total_patches, band_patch)
        torch_dataset_total = Data.TensorDataset(total_sf, y_tensor_total)
    else:
        raise ValueError(f"未知的输入模式: {spec.get('input_mode')}")

    total_iter = Data.DataLoader(torch_dataset_total, batch_size=batch_size, shuffle=False, num_workers=0)
    return total_iter


def HSI_MNF(X, MNF_ratio):
    denoised_bands = math.ceil(MNF_ratio * X.shape[-1])
    mnfr = spy.mnf(spy.calc_stats(X), spy.noise_from_diffs(X))
    denoised_data = mnfr.reduce(X, num=denoised_bands)

    return denoised_data


def data_pad_zero(data, patch_length):
    data_padded = np.lib.pad(data, ((patch_length, patch_length), (patch_length, patch_length), (0, 0)), 'constant',
                             constant_values=0)
    return data_padded

def img_show(x):
    spy.imshow(x)
    plt.show()


def index_assignment(index, row, col, pad_length):
    new_assign = {}  # dictionary.
    for counter, value in enumerate(index):
        assign_0 = value // col + pad_length
        assign_1 = value % col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign


def select_patch(data_padded, pos_x, pos_y, patch_length):
    selected_patch = data_padded[pos_x - patch_length:pos_x + patch_length + 1,
                     pos_y - patch_length:pos_y + patch_length + 1]
    return selected_patch


def select_vector(data_padded, pos_x, pos_y):
    select_vector = data_padded[pos_x, pos_y]
    return select_vector


def gain_neighborhood_pixel(mirror_image, point, idx, patch=5):
    x = point[idx, 0]
    y = point[idx, 1]
    return mirror_image[x:(x + patch), y:(y + patch), :]


def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    nn = band_patch // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch * patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch * patch * band_patch, band), dtype=float)
    center_start = nn * patch * patch
    x_train_band[:, center_start:(center_start + patch * patch), :] = x_train_reshape
    for i in range(nn):
        left_slice = x_train_reshape[:, :, band - i - 1:]
        left_wrap = x_train_reshape[:, :, :band - i - 1]
        x_train_band[:, i * patch * patch:(i + 1) * patch * patch, :i + 1] = left_slice
        x_train_band[:, i * patch * patch:(i + 1) * patch * patch, i + 1:] = left_wrap

        right_slice = x_train_reshape[:, :, i + 1:]
        right_wrap = x_train_reshape[:, :, :i + 1]
        start = (nn + i + 1) * patch * patch
        x_train_band[:, start:start + patch * patch, :band - i - 1] = right_slice
        x_train_band[:, start:start + patch * patch, band - i - 1:] = right_wrap
    return x_train_band


def HSI_create_pathes(data_padded, hsi_h, hsi_w, data_indexes, patch_length, flag):
    h_p, w_p, c = data_padded.shape

    data_size = len(data_indexes)
    patch_size = patch_length * 2 + 1

    data_assign = index_assignment(data_indexes, hsi_h, hsi_w, patch_length)
    if flag == 1:
        # for spatial net data, HSI patch
        unit_data = np.zeros((data_size, patch_size, patch_size, c))
        # 先创建在CPU上，避免一次性占用大量GPU显存
        unit_data_torch = torch.from_numpy(unit_data).type(torch.FloatTensor)
        for i in range(len(data_assign)):
            # 从GPU上的data_padded选择patch，然后移到CPU
            patch = select_patch(data_padded, data_assign[i][0], data_assign[i][1], patch_length)
            unit_data_torch[i] = patch.cpu()  # 移到CPU避免显存爆炸

    if flag == 2:
        # for spectral net data, HSI vector
        unit_data = np.zeros((data_size, c))
        # 先创建在CPU上，避免一次性占用大量GPU显存
        unit_data_torch = torch.from_numpy(unit_data).type(torch.FloatTensor)
        for i in range(len(data_assign)):
            # 从GPU上的data_padded选择vector，然后移到CPU
            vector = select_vector(data_padded, data_assign[i][0], data_assign[i][1])
            unit_data_torch[i] = vector.cpu()  # 移到CPU避免显存爆炸

    return unit_data_torch


def generate_data_set(data_reshape, label, index):
    train_data_index, test_data_index, all_data_index = index
    x_train_set = data_reshape[train_data_index]
    y_train_set = label[train_data_index] - 1

    x_test_set = data_reshape[test_data_index]
    y_test_set = label[test_data_index] - 1

    x_all_set = data_reshape[all_data_index]
    y_all_set = label[all_data_index] - 1

    return x_train_set, y_train_set, x_test_set, y_test_set, x_all_set, y_all_set

def train_and_test_data(mirror_image, band, train_point, test_point, true_point, patch=5, band_patch=3):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=float)
    for i in range(train_point.shape[0]):
        x_train[i,:,:,:] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j,:,:,:] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    for k in range(true_point.shape[0]):
        x_true[k,:,:,:] = gain_neighborhood_pixel(mirror_image, true_point, k, patch)
    print("x_train shape = {}, type = {}".format(x_train.shape,x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape,x_test.dtype))
    print("x_true  shape = {}, type = {}".format(x_true.shape,x_test.dtype))
    print("**************************************************")
    
    x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
    x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)
    x_true_band = gain_neighborhood_band(x_true, band, band_patch, patch)
    print("x_train_band shape = {}, type = {}".format(x_train_band.shape,x_train_band.dtype))
    print("x_test_band  shape = {}, type = {}".format(x_test_band.shape,x_test_band.dtype))
    print("x_true_band  shape = {}, type = {}".format(x_true_band.shape,x_true_band.dtype))
    print("**************************************************")
    return x_train_band, x_test_band, x_true_band

def generate_data_set_disjoint(data_reshape, label_train, label_test, index):
    train_data_index, test_data_index, all_data_index = index
    x_train_set = data_reshape[train_data_index]
    y_train_set = label_train[train_data_index] - 1

    x_test_set = data_reshape[test_data_index]
    y_test_set = label_test[test_data_index] - 1

    # x_all_set = data_reshape[all_data_index]
    # y_all_set = label[all_data_index] - 1

    return x_train_set, y_train_set, x_test_set, y_test_set


# generating HSI patches using GPU directly.
def generate_iter(data_padded, hsi_h, hsi_w, label_reshape, index, patch_length, batch_size,
                  model_type_flag,
                  model_3D_spa_flag, last_batch_flag):
    # flag for single spatial net or single spectral net or spectral-spatial net
    data_padded_torch = torch.from_numpy(data_padded).type(torch.FloatTensor).to(get_device())

    # for data label
    train_labels = label_reshape[index[0]] - 1
    test_labels = label_reshape[index[1]] - 1

    y_tensor_train = torch.from_numpy(train_labels).type(torch.FloatTensor)
    y_tensor_test = torch.from_numpy(test_labels).type(torch.FloatTensor)

    # for data
    if model_type_flag == 1:  # data for single spatial net
        spa_train_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[0], patch_length, 1)
        spa_test_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[1], patch_length, 1)

        if model_3D_spa_flag == 1:  # spatial 3D patch
            spa_train_samples = spa_train_samples.unsqueeze(1)
            spa_test_samples = spa_test_samples.unsqueeze(1)

        torch_dataset_train = Data.TensorDataset(spa_train_samples, y_tensor_train)
        torch_dataset_test = Data.TensorDataset(spa_test_samples, y_tensor_test)

    elif model_type_flag == 2:  # data for single spectral net
        spe_train_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[0], patch_length, 2)
        spe_test_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[1], patch_length, 2)

        torch_dataset_train = Data.TensorDataset(spe_train_samples, y_tensor_train)
        torch_dataset_test = Data.TensorDataset(spe_test_samples, y_tensor_test)

    elif model_type_flag == 3:  # data for spectral-spatial net
        # spatail data
        spa_train_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[0], patch_length, 1)
        spa_test_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[1], patch_length, 1)

        # spectral data
        spe_train_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[0], patch_length, 2)
        spe_test_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[1], patch_length, 2)

        torch_dataset_train = Data.TensorDataset(spa_train_samples, spe_train_samples, y_tensor_train)
        torch_dataset_test = Data.TensorDataset(spa_test_samples, spe_test_samples, y_tensor_test)

    if last_batch_flag == 0:
        train_iter = Data.DataLoader(dataset=torch_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
        test_iter = Data.DataLoader(dataset=torch_dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)
    elif last_batch_flag == 1:
        train_iter = Data.DataLoader(dataset=torch_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0,
                                     drop_last=True)
        test_iter = Data.DataLoader(dataset=torch_dataset_test, batch_size=batch_size, shuffle=False, num_workers=0,
                                    drop_last=True)
    # train_iter = Data.DataLoader(dataset=torch_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    # test_iter = Data.DataLoader(dataset=torch_dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_iter, test_iter


def generate_iter_disjoint(data_padded, hsi_h, hsi_w, gt_train_re, gt_test_re, index, patch_length, batch_size,
                           model_type_flag, model_3D_spa_flag):
    data_padded_torch = torch.from_numpy(data_padded).type(torch.FloatTensor).to(get_device())

    train_labels = gt_train_re[index[0]] - 1
    test_labels = gt_test_re[index[1]] - 1

    y_tensor_train = torch.from_numpy(train_labels).type(torch.FloatTensor)
    y_tensor_test = torch.from_numpy(test_labels).type(torch.FloatTensor)

    # for data
    if model_type_flag == 1:  # data for single spatial net
        spa_train_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[0], patch_length, 1)
        spa_test_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[1], patch_length, 1)

        if model_3D_spa_flag == 1:  # spatial 3D patch
            spa_train_samples = spa_train_samples.unsqueeze(1)
            spa_test_samples = spa_test_samples.unsqueeze(1)

        torch_dataset_train = Data.TensorDataset(spa_train_samples, y_tensor_train)
        torch_dataset_test = Data.TensorDataset(spa_test_samples, y_tensor_test)

    elif model_type_flag == 2:  # data for single spectral net
        spe_train_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[0], patch_length, 2)
        spe_test_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[1], patch_length, 2)

        torch_dataset_train = Data.TensorDataset(spe_train_samples, y_tensor_train)
        torch_dataset_test = Data.TensorDataset(spe_test_samples, y_tensor_test)

    elif model_type_flag == 3:  # data for spectral-spatial net
        # spatail data
        spa_train_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[0], patch_length, 1)
        spa_test_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[1], patch_length, 1)

        # spectral data
        spe_train_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[0], patch_length, 2)
        spe_test_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[1], patch_length, 2)

        torch_dataset_train = Data.TensorDataset(spa_train_samples, spe_train_samples, y_tensor_train)
        torch_dataset_test = Data.TensorDataset(spa_test_samples, spe_test_samples, y_tensor_test)

    train_iter = Data.DataLoader(dataset=torch_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_iter = Data.DataLoader(dataset=torch_dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_iter, test_iter


# all) generating HSI patches for the visualization of all the labeled samples of the data set
# total) generating HSI patches for the visualization of total the samples of the data set
# in addition, all) and total) both use GPU directly
def generate_iter_total(data_padded, hsi_h, hsi_w, label_reshape, index, patch_length, batch_size, model_type_flag,
                        model_3D_spa_flag):
    data_padded_torch = torch.from_numpy(data_padded).type(torch.FloatTensor).to(get_device())

    if len(index) < label_reshape.shape[0]:
        total_labels = label_reshape[index] - 1
    else:
        total_labels = np.zeros(label_reshape.shape)

    y_tensor_total = torch.from_numpy(total_labels).type(torch.FloatTensor)

    if model_type_flag == 1:
        total_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index, patch_length, 1)
        if model_3D_spa_flag == 1:  # spatial 3D patch
            total_samples = total_samples.unsqueeze(1)
        torch_dataset_total = Data.TensorDataset(total_samples, y_tensor_total)

    elif model_type_flag == 2:
        total_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index, patch_length, 2)
        torch_dataset_total = Data.TensorDataset(total_samples, y_tensor_total)
    elif model_type_flag == 3:
        spa_total_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index, patch_length, 1)
        spe_total_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index, patch_length, 2)
        torch_dataset_total = Data.TensorDataset(spa_total_samples, spe_total_samples, y_tensor_total)

    total_iter = Data.DataLoader(dataset=torch_dataset_total, batch_size=batch_size, shuffle=False, num_workers=0)

    return total_iter