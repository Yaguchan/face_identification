import os
from PIL import Image
from tqdm import tqdm


# python preprocess/data_resize.py
list_size = [32, 64, 128]
INDIR = './data/twitter/images/face'
OUTDIR = f'./data/twitter_resize/images/face'


def resize_image(input_path, output_path):
    with Image.open(input_path) as img:
        img.save(output_path.replace('.jpg', '_base.jpg'))
        for size in list_size:
            width, height = img.size
            if width > height:
                new_width = size
                new_height = int((size / width) * height)
            else:
                new_height = size
                new_width = int((size / height) * width)
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            resized_img.save(output_path.replace('.jpg', f'_{str(size)}.jpg'))


def main():
    member_names = os.listdir(INDIR)
    member_names = [member_name for member_name in member_names if member_name != '.DS_Store']
    for member_name in tqdm(member_names):
        member_dir = os.path.join(INDIR, member_name)
        output_dir = os.path.join(OUTDIR, member_name)
        os.makedirs(output_dir, exist_ok=True)
        img_names = os.listdir(member_dir)
        img_names = [img_name for img_name in img_names if img_name != '.DS_Store']
        for img_name in img_names:
            input_path = os.path.join(member_dir, img_name)
            output_path = os.path.join(output_dir, img_name)
            resize_image(input_path, output_path)



if __name__ == '__main__':
    main()