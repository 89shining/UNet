#----------------------------------------------------#
#  UNet 批量预测（医学图像 2D切片）
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
    weight_path = "logs/best_epoch_weights.pth"
    unet = Unet(backbone='resnet50')   # 与 Deeplab 一样

    # 自动兼容单/多GPU训练权重
    state_dict = torch.load(weight_path, map_location='cuda')
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v

    # ⚠️ 注意：Unet 类比 Deeplab 多封装一层
    # 实际 nn.Module 是 unet.net.module 或 unet.net.net
    # 所以需要尝试加载到正确的子模型中
    try:
        unet.net.net.load_state_dict(new_state_dict, strict=False)
        print(f"✅ 权重已加载到 unet.net.net")
    except Exception as e:
        try:
            unet.net.module.load_state_dict(new_state_dict, strict=False)
            print(f"✅ 权重已加载到 unet.net.module")
        except:
            unet.net.load_state_dict(new_state_dict, strict=False)
            print(f"✅ 权重已加载到 unet.net")

    print(f"已加载模型权重: {weight_path}")

    # 启用灰度mask输出（背景=0，目标=255）
    unet.gray_output = True

    #=============================================#
    #   输入/输出路径设置
    #=============================================#
    dir_origin_path = r"/home/wusi/UNet/img"      # 测试集切片
    dir_save_path   = r"/home/wusi/UNet/output"   # 结果保存路径
    ref_root        = r"/home/wusi/SAMdata/20250711_GTVp/datanii/test_nii"
    os.makedirs(dir_save_path, exist_ok=True)

    #=============================================#
    #   遍历测试集切片进行预测
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

        # 判断该层是否为空层
        gt_path = os.path.join(ref_root, pid, "GTVp.nii.gz")
        if not os.path.exists(gt_path):
            gt_path = os.path.join(ref_root, pid, "label.nii.gz")
        if not os.path.exists(gt_path):
            print(f"未找到GT: {pid}, 跳过空切片判断")
            gt_nonempty = True
        else:
            gt_vol = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
            if slice_id >= gt_vol.shape[0]:
                print(f"{pid} 第 {slice_id} 层超出范围，跳过")
                continue
            gt_nonempty = np.max(gt_vol[slice_id]) > 0

        # # 跳过空GT切片
        # if not gt_nonempty:
        #     continue

        #-----------------------------------------#
        # detect_image 输出二值mask (灰度)
        #-----------------------------------------#
        image = Image.open(image_path)
        r_image = unet.detect_image(image)

        save_path = os.path.join(dir_save_path, img_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        r_image.save(save_path)

    print(f"🎯 批量预测完成！结果保存在: {dir_save_path}")
