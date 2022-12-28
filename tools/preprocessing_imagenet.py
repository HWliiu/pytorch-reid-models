"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

from multiprocessing import Pool
from pathlib import Path

from PIL import Image


def resize_image(image_path):
    # 获取图片文件名和后缀名
    filename = image_path.stem
    ext = image_path.suffix
    # 获取图片所在的目录名
    dirname = image_path.parent
    # 创建新的目录名，用于保存resize后的图片
    new_dirname = (
        dirname.as_posix()
        .replace("train", "train_resized")
        .replace("val", "val_resized")
    )
    # 如果新的目录不存在，就创建它
    Path(new_dirname).mkdir(parents=True, exist_ok=True)
    # 创建新的文件名，用于保存resize后的图片
    new_image_path = Path(new_dirname) / (filename + ".jpg")
    # 打开原始图片，并resize成最长边为256的大小，并保持图像比例不变
    image = Image.open(image_path)
    image.thumbnail((256, 256))
    # 保存resize后的图片到新的文件名中
    image.save(new_image_path, quality=80)


if __name__ == "__main__":
    # 创建一个进程池，用于并行处理图片
    pool = Pool(20)

    # 获取imaget训练集和验证集的根目录（根据实际情况修改）
    root_dirs = [
        Path("/data/Datasets/ImageNet2012/train"),
        Path("/data/Datasets/ImageNet2012/val"),
    ]

    # 遍历每个根目录下的所有子目录和文件
    for root_dir in root_dirs:
        for image_path in root_dir.glob("**/*.JPEG"):
            # 对每个文件进行判断，如果是图片文件，就加入进程池中等待处理
            pool.apply_async(resize_image, args=(image_path,))

    # 关闭进程池，并等待所有任务完成
    pool.close()
    pool.join()
