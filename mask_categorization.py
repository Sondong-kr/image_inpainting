import os
import shutil
from PIL import Image
import numpy as np

def calculate_zero_percentage(image_path):
    """이미지 파일에서 0의 비율을 계산하는 함수"""
    image = Image.open(image_path)
    image_array = np.array(image)
    total_pixels = image_array.size
    zero_pixels = np.sum(image_array == 0)
    zero_percentage = (zero_pixels / total_pixels) * 100
    return zero_percentage

def sort_images_by_zero_percentage(dataset_folder, output_folder):
    """이미지를 0의 비율에 따라 분류하여 저장하는 함수"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(dataset_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        image_path = os.path.join(dataset_folder, filename)
        zero_percentage = calculate_zero_percentage(image_path)
        
        range_start = int(zero_percentage // 10) * 10
        range_end = range_start + 9
        folder_name = f"{range_start}~{range_end}%"
        folder_path = os.path.join(output_folder, folder_name)
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # 이미지 파일을 해당 폴더로 이동
        shutil.move(image_path, os.path.join(folder_path, filename))

# 실행 예제
dataset_folder = './mask/test_mask'
output_folder = './mask_category'
sort_images_by_zero_percentage(dataset_folder, output_folder)