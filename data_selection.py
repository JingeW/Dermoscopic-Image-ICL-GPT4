import os
import shutil
import pandas as pd
import argparse

def make_dir(path):
    """Create a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")
    else:
        print(f"Directory already exists: {path}")

def select_data(source, target_dir, ID_list):
    """Copy selected files from the source to the target directory."""
    for i in ID_list:
        src_path = os.path.join(source, i)
        if os.path.exists(src_path):
            shutil.copy(src_path, target_dir)
            print(f"Copied: {src_path} to {target_dir}")
        else:
            print(f"File not found: {src_path}")

def main(args):
    # Read selected data into data frame
    data_list_path = args.data_list  # Image names come from Image names come from the supplemental table 1 of DOI: 10.1016/j.jaad.2023.12.062
    df_mm = pd.read_excel(data_list_path, sheet_name=0, skiprows=1)
    df_bn = pd.read_excel(data_list_path, sheet_name=1, skiprows=1)

    # Extract image IDs and append file extension
    ID_mm = [f"{i}.jpg" for i in df_mm['ISIC ID'].to_list()]
    ID_bn = [f"{i}.jpg" for i in df_bn['ISIC ID'].to_exists_list()]
    ID_all = ID_mm + ID_bn

    # Create directories
    make_dir(args.target_mm)
    make_dir(args.target_bn)
    make_dir(args.target_all)

    # Select data from source
    select_data(args.source, args.target_mm, ID_mm)
    select_data(args.source, args.target_bn, ID_bn)
    select_data(args.source, args.target_all, ID_all)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Copy selected image data to specified directories.')
    parser.add_argument('--data_list', type=str, default='./data/selected_images.xlsx', help='Path to the Excel file containing image IDs.')
    parser.add_argument('--source', type=str, default='./RAW', help='Source directory where raw images are stored.')
    parser.add_argument('--target_mm', type=str, default='./data/mm', help='Target directory for melanoma images.')
    parser.add_argument('--target_bn', type=str, default='./data/bn', help='Target directory for benign nevus images.')
    parser.add_argument('--target_all', type=str, default='./data/all', help='Target directory for all selected images.')
    args = parser.parse_args()
    
    main(args)
