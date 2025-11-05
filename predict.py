#----------------------------------------------------#
#   UNet 批量预测（医学图像 2D切片）
#----------------------------------------------------#
import os
import torch
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from PIL import Image
from unet import Unet
from collections import OrderedDict
import re

if __name__ == "__main__":
    #=============================================#
    #   加载模型
    #=============================================#
    unet = Unet(backbone='resnet50')
    weight_path = "logs/best_epoch_weights.pth"

    # 自动兼容单/多GPU权重
    state_dict = torch.load(weight_path, map_location='cuda')
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    unet.net.load_state_dict(new_state_dict, strict=False)
    print(f"已加载模型权重: {weight_path}")

    # ✅ 启用灰度输出
    unet.gray_output = True

    #=============================================#
    #   路径设置
    #=============================================#
    dir_origin_path = r"/home/wusi/UNet/img"
    dir_save_path   = r"/home/wusi/UNet/output"
    ref_root        = r"/home/wusi/SAMdata/20250711_GTVp/datanii/test_nii"
    os.makedirs(dir_save_path, exist_ok=True)

    #=============================================#
    #   遍历预测
    #=============================================#
    img_names = [
        f for f in os.listdir(dir_origin_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))
    ]
    pattern = re.compile(r"^(p_\d+)_slice(\d+)\.(jpg|png|jpeg|tif)$", re.IGNORECASE)

    for img_name in tqdm(img_names, desc="Predicting"):
        image_path = os.path.join(dir_origin_path, img_name)
        match = pattern.match(img_name)
        if not match:
            continue
        pid = match.group(1)
        slice_id = int(match.group(2))

        gt_path = os.path.join(ref_root, pid, "GTVp.nii.gz")
        if not os.path.exists(gt_path):
            gt_path = os.path.join(ref_root, pid, "label.nii.gz")
        if not os.path.exists(gt_path):
            gt_nonempty = True
        else:
            gt_vol = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
            if slice_id >= gt_vol.shape[0]:
                continue
            gt_nonempty = np.max(gt_vol[slice_id]) > 0

        # # 跳过空切片
        # if not gt_nonempty:
        #     continue

        #-----------------------------------------#
        # 预测 & 保存
        #-----------------------------------------#
        image = Image.open(image_path)
        r_image = unet.detect_image(image)
        save_path = os.path.join(dir_save_path, img_name)
        r_image.save(save_path)

    print(f"🎯 批量预测完成！结果保存在: {dir_save_path}")
