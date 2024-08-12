from PIL import Image
import os
from .cal_porosity import calculate_porosity


# Calculate the weighted average and standard deviation, and output the reciprocal of the coefficient of variation as the final result
def calculate_weighted_avg_and_std(base_path, weights , file_name):
    # Ensure the weight list has a length of 10
    if len(weights) != 10:
        raise ValueError("The weight list must contain 10 elements.")

    # Read each '1.png' image in the sub-folders and calculate the porosity
    porosity_ratios = []
    for i in range(1, 11):
        sub_path = os.path.join(base_path, f'm{i}')
        try:
            image_path = os.path.join(sub_path, file_name)
            porosity_ratio = calculate_porosity(image_path)  # Assuming calculate_porosity is a predefined function
            porosity_ratios.append(porosity_ratio)
        except FileNotFoundError:
            print(f"The file at {image_path} was not found.")
            return None  # If the file is not found, return None

    # Calculate the weighted sum
    weighted_avg = sum(porosity_ratios[m - 1] * weights[m - 1] for m in range(1, 11))
    weighted_std = sum(((porosity_ratios[m - 1] - weighted_avg) ** 2) * weights[m - 1] for m in range(1, 11))
    wa = weighted_avg / len(weights)
    ws = weighted_std / len(weights)  # Calculate the avg and std
    return wa/ws