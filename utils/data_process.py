import numpy as np
import h5py
import cv2
from sklearn import preprocessing
import scipy.io as sio
import torch

def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
    return alpha * data + beta * noise

def patch(data, hsi_h, hsi_w, patch_length):
    h_slice = slice(hsi_h - patch_length, hsi_h + patch_length + 1)
    w_slice = slice(hsi_w - patch_length, hsi_w + patch_length + 1)
    patch = data[h_slice, w_slice, :]

    # patch = patch.reshape(-1, patch.shape[0] * patch.shape[1] * patch.shape[2])
    return patch

def standardization(data):
    height, width, bands = data.shape
    data = np.reshape(data, [height * width, bands])
    data = preprocessing.StandardScaler().fit_transform(data)

    data = np.reshape(data, [height, width, bands])
    return data
def generate_src_cross_Data(name_img, bands_random, patch_size):
    np.random.seed(123456789)

    if name_img == "BOT":
        img = sio.loadmat('../data/Botswana.mat')['Botswana'] # uint16 (1476, 256, 145)
        gt = sio.loadmat('../data/Botswana_gt.mat')['Botswana_gt'] # uint8 (1476, 256)
        pix_random_select_nonzero = 1286
        pix_nonzero = 3248
    elif name_img == "HU":
        img = sio.loadmat('../data/DFC2013_Houston.mat')['Houston'] # (349, 1905, 144)
        gt = sio.loadmat('../data/DFC2013_Houston_gt.mat')['Houston_gt'] # (349, 1905)
        pix_random_select_nonzero = 5948
        pix_nonzero = 15029
    elif name_img == "KSC":
        img = sio.loadmat('../data/KSC.mat')['KSC'] # (512, 614, 176)
        gt = sio.loadmat('../data/KSC_gt.mat')['KSC_gt'] # (512, 614)
        pix_random_select_nonzero = 2062
        pix_nonzero = 5211
    elif name_img == "CH":
        img = h5py.File('../data/Chikusei.mat')['chikusei'][:].transpose(1, 2, 0) # (2335, 2517, 128)
        gt = sio.loadmat('../data/Chikusei_gt.mat')['GT'][0][0][0] # (2517, 2335)
        pix_random_select_nonzero = 30704
        pix_nonzero = 77592

    img = img.astype(np.float32)
    m, n, b = img.shape
    # print(f"{name_img}: m={m}, n={n}, b={b}")
    img = standardization(img)
    # print(img.dtype)
    img = cv2.copyMakeBorder(img, patch_size, patch_size, patch_size, patch_size, cv2.BORDER_REFLECT)
    gt = gt.reshape(-1)
    mm, nn, bb = img.shape
    # print(f"{name_img}: gt_re.shape = {gt.shape}")
    # print(f"{name_img}: mm={mm}, nn={nn}, bb={bb}")

    indices = np.arange(pix_nonzero) # [0-77591]
    shuffled_indices = np.random.choice(indices, size=pix_random_select_nonzero, replace=False) # [选30704个]
    # data_division = [None] * pix_nonzero #77592
    count = 0
    num_group = 20

    # 默认64
    img_ = np.zeros([pix_random_select_nonzero, num_group, 2 * patch_size + 1, 2 * patch_size + 1, bands_random], dtype=np.float32)
    # gt_ = np.zeros([pix_random_select_nonzero], dtype=np.uint8)
    index = 0

    for i in range(patch_size, mm - patch_size):
        for j in range(patch_size, nn - patch_size):
            gt_index = (i - patch_size) * n + j - patch_size
            if gt[gt_index] != 0:
                if count in shuffled_indices: #
                    temp = patch(img, i, j, patch_size) # patch块 9*9*bands
                    data_list = [radiation_noise(temp[:, :, np.random.choice(bb, size=bands_random, replace=False)])for _ in range(num_group)] # [pat1,..pat20]
                    # data_division[count] = data_list
                    img_[index,:,:,:,:] = np.array(data_list)
                    index += 1
                count += 1
    # data_division = [item for item in data_division if item is not None]

    # data_division = np.array(data_division)
    # print(f"{name_img}: data_division.shape = {data_division.shape}")
    # 在不同机器上
    # file_path = '/root/autodl-tmp'
    # file_name = f'{name_img}_{pix_random_select_nonzero}_{num_group}_{patch_size * 2 + 1}_{patch_size * 2 + 1}_{bands_random}.h5'
    # full_file_path = os.path.join(file_path, file_name)
    # f = h5py.File(full_file_path, 'w')

    # 自己机器上
    f = h5py.File(f'../data/{name_img}_{img_.shape[0]}_{img_.shape[1]}_{img_.shape[2]}_{img_.shape[3]}_{img_.shape[4]}.h5','w')
    f['data'] = img_
    # gt = np.delete(gt, np.where(gt == 0)) - 1
    # print(name_img + ": " + str(np.unique(gt)), str(gt.shape))
    # f['label'] = gt
    f.close()


def generate_tar_FT_test_data(seed_number, name_img, bands_random, patch_size):

    if name_img == "IP":
        img = sio.loadmat('../data/Indian_pines_corrected.mat')['indian_pines_corrected']
        gt = sio.loadmat('../data/Indian_pines_gt.mat')['indian_pines_gt'].reshape(-1)
        pix_num = 10249
    elif name_img == "UP":
        img = sio.loadmat('../data/PaviaU.mat')['paviaU']
        gt = sio.loadmat('../data/PaviaU_gt.mat')['paviaU_gt'].reshape(-1)
        pix_num = 42776
    elif name_img == "PC":
        img = sio.loadmat('../data/Pavia.mat')['pavia']
        gt = sio.loadmat('../data/Pavia_gt.mat')['pavia_gt'].reshape(-1)
        pix_num = 148152
    elif name_img == "SA":
        img = sio.loadmat('../data/Salinas_corrected.mat')['salinas_corrected']
        gt = sio.loadmat('../data/Salinas_gt.mat')['salinas_gt'].reshape(-1)
        pix_num = 54129

    img = img.astype(np.float32)
    gt = gt.astype(np.uint8)
    m, n, b = img.shape
    # print(f"gt = {gt},\n gt.shape = {gt.shape},\n  np.unique(gt) = {np.unique(gt)}\n")

    gt_ = []
    for each in gt:
        if each != 0:
            gt_.append(each)
    gt_ = np.array(gt_)
    cls_count = gt_.max()
    # print(f"gt_ = {gt_},\n gt_.shape = {gt_.shape},\n  np.unique(gt_) = {np.unique(gt_)}\n")
    # print("gt_, gt_.shape, np.unique(gt_)", gt_, gt_.shape, np.unique(gt_))

    # patch_size = 16
    img = standardization(img)
    # cv2.BORDER_REFLECT-边界元素的镜像方式
    img = cv2.copyMakeBorder(img, patch_size, patch_size, patch_size, patch_size, cv2.BORDER_REFLECT)
    [mm, nn, bb] = img.shape
    # num_group = 10  # 挑十次.
    # 64
    # bands_random = 6
    #train_dataset = [None] * gt_.shape[0]
    train_dataset = np.zeros([pix_num, 2 * patch_size + 1, 2 * patch_size + 1, bands_random],dtype=np.float32)
    index = 0
    for i in range(patch_size, mm - patch_size):
        for j in range(patch_size, nn - patch_size):
            gt_index = (i - patch_size) * n + j - patch_size
            if gt[gt_index] != 0:
                temp = patch(img, i, j, patch_size)[:, :, np.random.choice(bb, size=bands_random, replace=False)]
                # data_list = [temp[:, :, np.random.choice(bb, size=bands_random, replace=False)] for _ in range(num_group)]  # 10 29 29 6
                # data_list = [temp[:, :, np.random.choice(bb, size=bands_random, replace=False)]]  # 10 29 29 6
                # data_division.append(data_list) # 10 29 29 6
                train_dataset[index] = temp   # 个 29 29 6
                index += 1

    # train_dataset = np.array(train_dataset)  # 10249 29 29 6
    test_dataset = train_dataset # 10249 9 9 64
    # test_dataset = train_dataset.transpose(1, 0, 2, 3, 4)  # 10 10249 29 29 6 我不用投票机制

    f = h5py.File(
        f'../data/{name_img}_test_{test_dataset.shape[0]}_{test_dataset.shape[1]}_{test_dataset.shape[2]}_{test_dataset.shape[3]}.h5', 'w')
    f['data'] = test_dataset
    f['label'] = gt_ - 1
    f.close()

    # -----------------------------------------------------------------#

    np.random.seed(int(seed_number))

    indices = np.arange(train_dataset.shape[0])
    shuffled_indices = np.random.permutation(indices)

    train_dataset = train_dataset[shuffled_indices] # 10249 9 9 64
    gt_ = gt_[shuffled_indices]
    sample_preclass = 5
    data = np.zeros([sample_preclass*cls_count, 2 * patch_size + 1, 2 * patch_size + 1, bands_random], dtype=np.float32)
    index_data = 0


    for class_index in range(cls_count):  # 类别0-8
        count = 0
        for index in range(train_dataset.shape[0]):  # 数量0-4
            if gt_[index] == class_index + 1 and count < sample_preclass:
                # data.append(train_dataset[index])
                data[index_data,:,:,:] = train_dataset[index]
                count += 1
                index += 1
    # data = np.array(data).transpose(1, 0, 2, 3, 4)
    gt = np.array(range(cls_count))[:, np.newaxis]
    gt = np.repeat(gt, sample_preclass, axis=1).reshape(-1)
    #
    f = h5py.File(f'../data/{name_img}_FN_{data.shape[0]}_{data.shape[1]}_{data.shape[2]}_{data.shape[3]}_{sample_preclass}sample_preclass_{seed_number}seed_perc.h5', 'w')
    f['data'] = data  # 80 9 9 64
    f['label'] = gt  # (80,)
    f.close()




