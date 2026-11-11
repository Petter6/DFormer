from .. import *

# Dataset config
"""Dataset Path"""
C.dataset_name = os.environ.get(
    "DATASET", "DeStijl"
)  # default als de variabele niet gezet is
C.num_train_imgs = int(os.environ.get(
    "DS_SIZE", 100
)) 
C.batch_size = int(os.environ.get(
    "BS", 20
))  # default als de variabele niet gezet is
C.lr = float(os.environ.get(
    "LR", 1e-4
))  
C.dataset_path = osp.join(C.root_dir, C.dataset_name)
C.rgb_root_folder = osp.join(C.dataset_path, "RGB")
C.rgb_format = ".png"
C.gt_root_folder = osp.join(C.dataset_path, "Label")
C.gt_format = ".png"
C.gt_transform = True
# True when label 0 is invalid, you can also modify the function _transform_gt in dataloader.RGBXDataset
# True for most dataset valid, Faslse for MFNet(?)
C.x_root_folder = osp.join(C.dataset_path, "Depth")
C.x_format = ".png"
C.x_is_single_channel = (
    True  # True for raw depth, thermal and aolp/dolp(not aolp/dolp tri) input
)
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.is_test = True
C.num_eval_imgs = 300
C.num_classes = 11
C.class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "bg"]

"""Image Config"""
C.background = 255
C.image_height = 224
C.image_width = 224
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])
