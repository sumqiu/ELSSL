import math

from .cal_coefficients import calculate_iou,calculate_dice


# Calculate the IoU for a single model on the validation set
def calculate_Wi_iou(path1, path2, n):
    iou = []
    for i in range(1, n+1):
        file_name1 = "val" + str(i) + "_unet_seg.png"
        file_name2 = str(i) + ".png"
        iou.append(calculate_iou(path1, file_name1, path2, file_name2))
    return sum(iou) / len(iou)


# Calculate the Dice for a single model on the validation set
def calculate_Wi_dice(path1, path2, n):
    dice = []
    for i in range(1, n+1):  # n is the number of images in the val folder
        file_name1 = "val" + str(i) + "_unet_seg.png"
        file_name2 = str(i) + ".png"
        dice.append(calculate_dice(path1, file_name1, path2, file_name2))
    return sum(dice) / len(dice)


def calculate_Wi2(a, b, c, path1, path2, m, n):
    wi = []
    wi_avg = []
    wi_l2 = []
    p = []  # Iou
    q = []  # dice

    for i in range(m):
        p.append(calculate_Wi_iou(path1, path2 + "/m" + str(i+1), n))
        q.append(calculate_Wi_dice(path1, path2 + "/m" + str(i+1), n))
        x = a * (b * p[i] + (1 - b) * q[i])
        wi.append(math.exp(x))
    for j in range(m):
        wi_avg.append(wi[j] / sum(wi))

    w_all = c * sum(x**2 for x in wi_avg)
    # L2 regularization
    for x in wi_avg:
        wi_l2.append(x / math.sqrt(1 + w_all))
    return check_w(wi_l2)


def check_w(values):
    min_value = min(values)
    max_value = max(values)
    normalized_1 = [(value - min_value) / (max_value - min_value) for value in values]
    normalized = [0.2 + 0.3 * value for value in normalized_1]
    return normalized
