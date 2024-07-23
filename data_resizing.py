import os
import argparse
from PIL import Image, UnidentifiedImageError

def make_dir(path):
    """Ensure the directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def resize_image(image_path, save_dir, max_size):
    """Resize an image maintaining the aspect ratio."""
    try:
        with Image.open(image_path) as image:
            original_size = image.size
            ratio = min(max_size[0] / original_size[0], max_size[1] / original_size[1])
            new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
            resized_image = image.resize(new_size, Image.LANCZOS)
            image_name = os.path.splitext(os.path.basename(image_path))[0] + '.jpg'
            save_path = os.path.join(save_dir, image_name)
            resized_image.save(save_path)
            print(f"Resized and saved: {save_path}")
    except UnidentifiedImageError:
        print(f"Failed to identify or open image: {image_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def process_images(source_dir, target_dir, max_size):
    """Process all images in the directory."""
    make_dir(target_dir)
    for image_name in os.listdir(source_dir):
        if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(source_dir, image_name)
            resize_image(image_path, target_dir, max_size)
        else:
            print(f"Skipped non-image file: {image_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize images while maintaining aspect ratio.")
    parser.add_argument('--source_dir', type=str, default='./data/mm', help='Source directory of images.')
    parser.add_argument('--target_dir', type=str, default='./data/mm_resized', help='Target directory for resized images.')
    parser.add_argument('--max_width', type=int, default=500, help='Maximum width of the resized images.')
    parser.add_argument('--max_height', type=int, default=500, help='Maximum height of the resized images.')
    args = parser.parse_args()

    max_size = (args.max_width, args.max_height)
    process_images(args.source_dir, args.target_dir, max_size)
