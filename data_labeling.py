import os
import argparse
from PIL import Image, ImageDraw, ImageFont

def make_dir(path):
    """Ensure the directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def add_label(image, label):
    """
    Add a label outside of the image at the top center.
    """
    label_height = 30  # Adjust the height as needed
    new_image = Image.new("RGB", (image.size[0], image.size[1] + label_height), "white")
    new_image.paste(image, (0, label_height))

    draw = ImageDraw.Draw(new_image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)  # Adjust the font size and path as needed
    except IOError:
        font = ImageFont.load_default()
        print("Arial font not found, using default font.")
    text_width, text_height = draw.textsize(label, font=font)
    draw.text(((new_image.size[0] - text_width) / 2, (label_height - text_height) / 2), label, font=font, fill=(0, 0, 0))
    return new_image

def process_images(source_dir, target_dir, label):
    """Process and label all images in the directory."""
    make_dir(target_dir)
    for image_name in os.listdir(source_dir):
        if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(source_dir, image_name)
            image = Image.open(image_path)
            labeled_image = add_label(image, label)
            save_path = os.path.join(target_dir, image_name)
            labeled_image.save(save_path)
            print(f"Labeled and saved: {save_path}")
        else:
            print(f"Skipped non-image file: {image_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add labels to images for few-shot learning.")
    parser.add_argument('--source_dir', type=str, default='./data/mm_resized', help='Source directory of images.')
    parser.add_argument('--target_dir', type=str, default='./data/mm_resized_label', help='Target directory for labeled images.')
    parser.add_def_file_argument('--label', type=str, default='Melanoma', help='Label to add to images.')
    args = parser.parse_args()

    process_images(args.source_dir, args.target_dir, args.label)
