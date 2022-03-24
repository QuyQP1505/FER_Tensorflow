import os
from PIL import Image

train_path = r'D:\Machine_Learning\FEC - version 2\data\train_class'
test_path = r'D:\Machine_Learning\FEC - version 2\data\val_class'

img_size = 48

def resize_data(path, size):
    count = 0
    dirs = os.listdir(path)  # Location of path
    for category in dirs:
        temp_path = os.path.join(path, category)
        for item in os.listdir(temp_path):
            img_path = os.path.join(temp_path, item)
            if os.path.isfile(img_path):

                im = Image.open(img_path)

                # Split path
                split_file = os.path.split(img_path)
                split_folder = os.path.split(split_file[0])
                path_name = os.path.join(split_folder[0] + "_resize",
                                         split_folder[1] + "_resize")

                # Convert and resize image
                img_resize = im.resize((size, size), Image.ANTIALIAS)
                img_resize = img_resize.convert('L')

                # Save image to new location (Notice: Need to pre-create category folder)
                save_path = os.path.join(path_name, split_file[1])
                img_resize.save(save_path, 'JPEG')

                # Display info
                print("[INFO] Saving into:" + save_path)
                count += 1

            else:
                # Stop when found error
                print("[INFO] Found something is not a image file!")
                break

    print("[INFO] Total of image resized: " + str(count))

# AFTER RESIZE AffectNET Kaggle NOT Original dataset:
# - Train dataset: 33803
# - Test dataset: 4000
