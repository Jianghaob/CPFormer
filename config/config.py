import os
from re import T


EXPERIMENT_CONFIG = {
    "model_name": "CPFormer", 
    "dataset": {
        "name": "IP",
        "data_path": os.path.join(os.getcwd(), "data"),
        "patch_size": 13,
        "band_patch": 3,
        "pca_components": 120,
        # 数据划分方式：ratio 按比例；fixed 按每类固定数量
        "sampling_mode": "ratio",   
        # 是否启用验证集（若为 False，则只划分训练/P测试）
        "use_validation": True,
        # ratio 
        "train_ratio": 0.01,
        "val_ratio": 0.01, # use_validation=True
        # fixed 
        "train_num_per_class": 10,
        "val_num_per_class": 5,# use_validation=True
    },
    "training": {
        "batch_size": 32,
        "max_epoch": 150, 
        "learning_rate": 0.001, 
        "depth": 2, 
        "heads": 8,
        "mlp_dim": 128,
        "dropout": 0.1,
        "dim": 256, # UP、HU13、HU18：64, IP:256
        "num_groups": 4,
        "sf_mode": "CAF",
        "sf_dim_head": 8,
        "sf_dropout": 0.1,
        "sf_emb_dropout": 0.1,
        "seed_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    },
}


MODEL_SPECS = {
    "CPFormer": {
        "input_mode": "spa",
        "use_pca_patch": True,
        "needs_3d": False,
    },
    "CVSSN": {
        "input_mode": "spa_spe",
        "use_pca_patch": False,
        "vector_use_pca": False,
        "needs_3d": False,
    },
    "SQSFormer": {
        "input_mode": "spa",
        "use_pca_patch": True,
        "needs_3d": False,
    },
    "DBDA": {
        "input_mode": "spa",
        "use_pca_patch": False,
        "needs_3d": True,
    },
    "SSFTT": {
        "input_mode": "spa",
        "use_pca_patch": True,
        "needs_3d": True,
    },
    "SF": {
        "input_mode": "sf",
        "use_pca_patch": False,
        "needs_3d": False,
    },
    "S2FTNet": { # ps=13, 5e-3、64、pca=30
        "input_mode": "spa_spe",
        "use_pca_patch": True,
        "vector_use_pca": False,
        "needs_3d": True,
    },
    "HMSSF": {
        "input_mode": "spa",
        "use_pca_patch": True,
        "needs_3d": True,
    },
    "CWSST": {
        "input_mode": "spa",
        "use_pca_patch": True,
        "needs_3d": False,
    }
}

