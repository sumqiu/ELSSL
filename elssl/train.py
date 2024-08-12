import subprocess

from pre_segmentation import pre_seg
from dataloader.process import *
from utils.cal_wa_std import *
from utils.cal_w import *

base_dir_mp = "models/"
file_name = "best_weight.pth"
base_dir_rp = "outs/pre_seg_imgs/"
path1 = "outs/val_seg_imgs"
s_path = "outs/synthetic_seg_imgs"
names = ["1_2000"]
m = 10
n = len(os.listdir(path1))

subprocess.run(['python', 'pre_train.py'], check=True)
val_process()

for item in names:
    for i in range(1, 11):
        mp = base_dir_mp + "m" + str(i) + '/' + file_name
        rp = base_dir_rp + "m" + str(i)
        if not os.path.exists(rp):
            os.makedirs(rp)
        pre_seg(mp, rp, "data/val")

    w = calculate_Wi2(20, 0.5, 1, path1, base_dir_rp, m, n)
    syn_process(item)
    for j in range(1, 11):
        mp = base_dir_mp + "m" + str(j) + '/' + file_name
        rp = s_path + "/m" + str(j)
        if not os.path.exists(rp):
            os.makedirs(rp)
        pre_seg(mp, rp, "data/synthetic/" + item)
    delete_mask_files(item)

    for i in range(1, 6):
        cx = calculate_weighted_avg_and_std(s_path, w, str(i) + ".png")
        if cx > 20:
            max_index = w.index(max(w))
            source_path = f"data/synthetic/{item}/{str(i - 1)}.png"
            target_path = f"{s_path}/m{str(max_index + 1)}/{str(i)}.png"
            train_process(source_path, target_path)

