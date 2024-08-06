from PIL import Image


def calculate_porosity(image_path):
    # Open the image file
    with Image.open(image_path) as img:
        # Convert the image to grayscale mode, if it is already a binary image, this step can be omitted
        img = img.convert('L')

        # Get the width and height of the image
        width, height = img.size

        # Calculate the total number of pixels in the image
        total_pixels = width * height

        # Initialize the counter for the black pixels
        black_pixel_count = 0

        # Traverse each pixel in the image
        for pixel in img.getdata():
            # If the pixel value is less than a certain threshold (e.g., 128), it is considered a pore
            if pixel < 128:
                black_pixel_count += 1

        # Calculate the ratio of the area of the pores
        black_ratio = black_pixel_count / total_pixels

        return black_ratio