import os
import random
import shutil

import cv2


def val_process():  # Set the path for the source folder and the destination folder
    src_folder = "./data/val"
    dst_folder = "./outs/val_seg_imgs"
    # Iterate through all files in the source folder
    for filename in sorted(os.listdir(src_folder)):
        # Check if the file ends with '_mask.png'
        if filename.endswith('_mask.png'):
            # Extract the prefix part of the filename
            prefix = filename.split('_mask')[0]

            # Convert the prefix to an integer
            number = int(prefix)

            # Increment the number by 1 to get the new number
            new_number = number + 1

            # Construct a new filename, maintaining the original number of digits
            new_filename = f'val{str(new_number)}_unet_seg.png'

            # Construct the full path for the source file and the destination file
            src_file = os.path.join(src_folder, filename)
            dst_file = os.path.join(dst_folder, new_filename)

            # Copy the file
            shutil.copy(src_file, dst_file)  # This line indicates that the file has been copied to the new filename


def train_process(src_path, src_mask_path):
    train_path = "./data/p_label"
    new_file_name = str(len(os.listdir(train_path)) // 2) + ".png"
    new_mask_file_name = str(len(os.listdir(train_path)) // 2) + "_mask.png"
    shutil.copy(src_path, train_path + "/" + new_file_name)
    shutil.copy(src_mask_path, train_path + "/" + new_mask_file_name)


def random_train(path):
    items = os.listdir(path)
    half = len(items) // 2
    random_index = random.randint(0, half - 1)
    return random_index


def syn_process(f_name):
    source_dir = './data/synthetic/' + f_name
    target_dir = './data/synthetic/' + f_name
    for filename in os.listdir(source_dir):
        if filename.endswith('.png'):
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, filename.replace('.png', '_mask.png'))
            image = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
            _, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cv2.imwrite(target_path, thresholded_image)


def delete_mask_files(f_name):
    directory = './data/synthetic/' + f_name
    for filename in os.listdir(directory):
        if filename.endswith('_mask.png'):
            file_path = os.path.join(directory, filename)
            os.remove(file_path)


# Once the article is published, we will update this function's code
def train_batch_size(i):
    return 1


def train_num_epochs():
    return random.randint(20, 30)


def train_path(i):
    train_path = "data/train"
    new_train_path = "models/m" + str(i) + "/train"
    if os.path.exists(new_train_path):
        shutil.rmtree(new_train_path)
    os.makedirs(new_train_path)

    n = len(os.listdir(train_path)) // 2
    for i in range(int(n/2)):
        x = random_train(train_path)
        src_image_path = os.path.join(train_path, str(x) + '.png')
        src_mask_path = os.path.join(train_path, str(x) + '_mask.png')
        dest_image_path = os.path.join(new_train_path, str(i) + '.png')
        dest_mask_path = os.path.join(new_train_path, str(i) + '_mask.png')

        shutil.copy(src_image_path, dest_image_path)
        shutil.copy(src_mask_path, dest_mask_path)
    return new_train_path
