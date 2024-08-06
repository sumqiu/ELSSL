import os
import shutil


def val_process():
    # Set the path for the source folder and the destination folder
    src_folder = "../data/val"
    dst_folder = "../outs/val_seg_imgs"

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
            shutil.copy(src_file, dst_file)
            print(f'Copied {filename} to {new_filename}')  # This line indicates that the file has been copied to the new filename


def train_process(src_path, src_mask_path):
    train_path = "../data/train"
    new_file_name = str(len(os.listdir(train_path)) // 2) + ".png"
    new_mask_file_name = str(len(os.listdir(train_path)) // 2) + "_mask.png"
    shutil.copy(src_path, train_path + "/" + new_file_name)
    shutil.copy(src_mask_path, train_path + "/" + new_mask_file_name)


# Once the article is published, we will update this function's code
def train_batch_size(i):
    return 1


def train_num_epochs(i):
    return 20


def train_path(i):
    return "data/train"
