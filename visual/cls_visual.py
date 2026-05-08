import torch
import numpy as np
import spectral as spy
from spectral import spy_colors

spy.algorithms

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


def gt_cls_map(gt_hsi, path):
    spy.save_rgb(path + "_gt.png", gt_hsi, colors=spy_colors)
    print('------Get ground truth classification map successful-------')


def pred_cls_map_dl(sample_list, net, gt_hsi, path, model_type_flag): 
    pred_sample = []
    pred_label = []

    net.eval()
    if len(sample_list) == 1:
        iter = sample_list[0]
        if model_type_flag == 1:  # data for single spatial net
            for X_spa, y in iter:
                X_spa = X_spa.to(get_device())
                y_pred = net(X_spa)
                if isinstance(y_pred, (tuple, list)):
                    y_pred = y_pred[0]
                # 确保y_pred是2维张量 (batch_size, num_classes)
                if y_pred.dim() == 1:
                    y_pred = y_pred.unsqueeze(0)
                pre_y = y_pred.cpu().argmax(axis=1).detach().numpy()
                pred_sample.extend(pre_y + 1) # 存的是所有样本
        elif model_type_flag == 2:  # data for single spectral net
            for X_spe, y in iter:
                X_spe = X_spe.to(get_device())
                y_pred = net(X_spe)
                if isinstance(y_pred, (tuple, list)):
                    y_pred = y_pred[0]
                # 确保y_pred是2维张量 (batch_size, num_classes)
                if y_pred.dim() == 1:
                    y_pred = y_pred.unsqueeze(0)
                pre_y = y_pred.cpu().argmax(axis=1).detach().numpy()
                pred_sample.extend(pre_y + 1)
        elif model_type_flag == 3:
            for X_spa, X_spe, y in iter:
                X_spa, X_spe, y = X_spa.to(get_device()), X_spe.to(get_device()), y.to(get_device())
                y_pred = net(X_spa, X_spe)
                if isinstance(y_pred, (tuple, list)):
                    y_pred = y_pred[0]
                # 确保y_pred是2维张量 (batch_size, num_classes)
                if y_pred.dim() == 1:
                    y_pred = y_pred.unsqueeze(0)
                pre_y = y_pred.cpu().argmax(axis=1).detach().numpy()
                pred_sample.extend(pre_y + 1)
    elif len(sample_list) == 2:
        iter, index = sample_list[0], sample_list[1]
        # print(len(index)) # 5211
        if model_type_flag == 1:  # data for single spatial net
            for X_spa, y in iter:
                X_spa = X_spa.to(get_device())
                y_pred = net(X_spa)
                if isinstance(y_pred, (tuple, list)):
                    y_pred = y_pred[0]
                # 确保y_pred是2维张量 (batch_size, num_classes)
                if y_pred.dim() == 1:
                    y_pred = y_pred.unsqueeze(0)
                pre_y = y_pred.cpu().argmax(axis=1).detach().numpy()
                pred_label.extend(pre_y + 1)
        elif model_type_flag == 2:  # data for single spectral net
            for X_spe, y in iter:
                X_spe = X_spe.to(get_device())
                y_pred = net(X_spe)
                if isinstance(y_pred, (tuple, list)):
                    y_pred = y_pred[0]
                # 确保y_pred是2维张量 (batch_size, num_classes)
                if y_pred.dim() == 1:
                    y_pred = y_pred.unsqueeze(0)
                pre_y = y_pred.cpu().argmax(axis=1).detach().numpy()
                pred_label.extend(pre_y + 1)
        elif model_type_flag == 3:
            for X_spa, X_spe, y in iter:
                X_spa, X_spe, y = X_spa.to(get_device()), X_spe.to(get_device()), y.to(get_device())
                y_pred = net(X_spa, X_spe)
                if isinstance(y_pred, (tuple, list)):
                    y_pred = y_pred[0]
                # 确保y_pred是2维张量 (batch_size, num_classes)
                if y_pred.dim() == 1:
                    y_pred = y_pred.unsqueeze(0)
                pre_y = y_pred.cpu().argmax(axis=1).detach().numpy()
                pred_label.extend(pre_y + 1)

        gt = np.ravel(gt_hsi)
        pred_sample = np.zeros(gt.shape)
        #print(f"pred_sample.shape={pred_sample.shape},pred_label={len(pred_label)}")
        pred_sample[index] = pred_label

    pred_hsi = np.reshape(pred_sample, (gt_hsi.shape[0], gt_hsi.shape[1]))
    spy.save_rgb(path + '_' + str(len(sample_list)) + '_pre.png', pred_hsi, colors=spy_colors)  # dpi haven't set now
    print('------Get pred classification maps successful-------')


def pred_cls_map_cls(sample_list, gt_hsi, path):
    if len(sample_list) == 1:
        pred_sample = sample_list[0]

    elif len(sample_list) == 2:
        pred_label, index = sample_list[0], sample_list[1]
        gt = np.ravel(gt_hsi)
        pred_sample = np.zeros(gt.shape)
        pred_sample[index] = pred_label

    pred_hsi = np.reshape(pred_sample, (gt_hsi.shape[0], gt_hsi.shape[1]))
    spy.save_rgb(path + '_' + str(len(sample_list)) + '_pre.png', pred_hsi, colors=spy_colors)  # dpi haven't set now
    print('------Get pred classification maps successful-------')
