"""
Convert IMDB-WIKI dataset from .mat format to label.txt format
This script reads the original .mat files and generates label.txt files
that match the expected format for AgeGenderDataset.
"""

import os
import numpy as np
from scipy.io import loadmat
import argparse


def convert_imdb_mat_to_txt(mat_file_path, output_label_path, image_base_path):
    """
    Convert IMDB .mat file to label.txt format
    
    Args:
        mat_file_path: Path to imdb.mat file
        output_label_path: Path to output label.txt file
        image_base_path: Base path where images are stored (relative to label.txt)
    """
    print(f"Loading {mat_file_path}...")
    mat_data = loadmat(mat_file_path)
    
    # Extract data from .mat file
    # IMDB .mat file structure:
    # - dob: date of birth (in days since 0000-01-01)
    # - photo_taken: year when photo was taken
    # - full_path: relative path to image
    # - gender: 0=female, 1=male, NaN=unknown
    # - face_location: face bounding box
    # - face_score: face detection score
    # - second_face_score: second face detection score
    
    dob = mat_data['dob'][0]  # Date of birth
    photo_taken = mat_data['photo_taken'][0]  # Year photo was taken
    full_path = mat_data['full_path'][0]  # Image paths
    gender = mat_data['gender'][0]  # Gender labels
    
    print(f"Found {len(dob)} images")
    
    # Calculate age and prepare labels
    valid_count = 0
    with open(output_label_path, 'w') as f:
        # Write header
        f.write("image_path age gender\n")
        
        for i in range(len(dob)):
            # Calculate age
            if not np.isnan(dob[i]) and not np.isnan(photo_taken[i]):
                # Convert dob from days to year
                birth_year = int(dob[i] / 365.25)
                photo_year = int(photo_taken[i])
                age = photo_year - birth_year
                
                # Get gender (0=female, 1=male, NaN=unknown)
                if not np.isnan(gender[i]):
                    gender_val = int(gender[i])
                else:
                    gender_val = np.nan
                
                # Get image path
                img_path = full_path[i][0] if isinstance(full_path[i][0], str) else str(full_path[i][0])
                
                # Only write valid entries (age > 0 and gender is known)
                if age > 0 and not np.isnan(gender_val):
                    # Write: image_path age gender
                    # gender: 0.0 for female, 1.0 for male
                    f.write(f"{img_path} {age:.1f} {gender_val:.1f}\n")
                    valid_count += 1
    
    print(f"Generated {output_label_path} with {valid_count} valid entries")


def convert_wiki_mat_to_txt(mat_file_path, output_label_path, image_base_path):
    """
    Convert WIKI .mat file to label.txt format
    
    Args:
        mat_file_path: Path to wiki.mat file
        output_label_path: Path to output label.txt file
        image_base_path: Base path where images are stored (relative to label.txt)
    """
    print(f"Loading {mat_file_path}...")
    mat_data = loadmat(mat_file_path)
    
    # Extract data from .mat file (similar structure to IMDB)
    dob = mat_data['dob'][0]
    photo_taken = mat_data['photo_taken'][0]
    full_path = mat_data['full_path'][0]
    gender = mat_data['gender'][0]
    
    print(f"Found {len(dob)} images")
    
    valid_count = 0
    with open(output_label_path, 'w') as f:
        # Write header
        f.write("image_path age gender\n")
        
        for i in range(len(dob)):
            # Calculate age
            if not np.isnan(dob[i]) and not np.isnan(photo_taken[i]):
                birth_year = int(dob[i] / 365.25)
                photo_year = int(photo_taken[i])
                age = photo_year - birth_year
                
                # Get gender
                if not np.isnan(gender[i]):
                    gender_val = int(gender[i])
                else:
                    gender_val = np.nan
                
                # Get image path
                img_path = full_path[i][0] if isinstance(full_path[i][0], str) else str(full_path[i][0])
                
                # Only write valid entries
                if age > 0 and not np.isnan(gender_val):
                    f.write(f"{img_path} {age:.1f} {gender_val:.1f}\n")
                    valid_count += 1
    
    print(f"Generated {output_label_path} with {valid_count} valid entries")


def setup_imdb_wiki_structure(imdb_crop_path, wiki_crop_path, output_base_path):
    """
    Set up the directory structure and convert IMDB-WIKI datasets
    
    Args:
        imdb_crop_path: Path to imdb_crop directory
        wiki_crop_path: Path to wiki_crop directory (if exists)
        output_base_path: Base path for output (should match config.age_gender_data_path)
    """
    # Create directory structure
    imdb_output_path = os.path.join(output_base_path, "IMDB")
    wiki_output_path = os.path.join(output_base_path, "WIKI")
    
    os.makedirs(imdb_output_path, exist_ok=True)
    os.makedirs(os.path.join(imdb_output_path, "data"), exist_ok=True)
    
    if wiki_crop_path and os.path.exists(wiki_crop_path):
        os.makedirs(wiki_output_path, exist_ok=True)
        os.makedirs(os.path.join(wiki_output_path, "data"), exist_ok=True)
    
    # Convert IMDB
    imdb_mat_file = os.path.join(imdb_crop_path, "imdb.mat")
    if os.path.exists(imdb_mat_file):
        imdb_label_file = os.path.join(imdb_output_path, "label.txt")
        convert_imdb_mat_to_txt(imdb_mat_file, imdb_label_file, imdb_crop_path)
        
        # Create symlink or copy images to data directory
        # For now, we'll use the original path structure
        print(f"\nIMDB conversion complete!")
        print(f"Label file: {imdb_label_file}")
        print(f"Note: Make sure image paths in label.txt match the actual image locations")
    else:
        print(f"Warning: {imdb_mat_file} not found!")
    
    # Convert WIKI
    if wiki_crop_path and os.path.exists(wiki_crop_path):
        wiki_mat_file = os.path.join(wiki_crop_path, "wiki.mat")
        if os.path.exists(wiki_mat_file):
            wiki_label_file = os.path.join(wiki_output_path, "label.txt")
            convert_wiki_mat_to_txt(wiki_mat_file, wiki_label_file, wiki_crop_path)
            print(f"\nWIKI conversion complete!")
            print(f"Label file: {wiki_label_file}")
        else:
            print(f"Warning: {wiki_mat_file} not found!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert IMDB-WIKI .mat files to label.txt format')
    parser.add_argument('--imdb_crop', type=str, 
                       default='/workspace/SwinFace/swinface_project/dataset/imdb-wiki/imdb_crop',
                       help='Path to imdb_crop directory')
    parser.add_argument('--wiki_crop', type=str, default=None,
                       help='Path to wiki_crop directory (optional)')
    parser.add_argument('--output', type=str,
                       default='/workspace/SwinFace/swinface_project/data/age_gender',
                       help='Output base path (should match config.age_gender_data_path)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("IMDB-WIKI Dataset Converter")
    print("=" * 60)
    print(f"IMDB crop path: {args.imdb_crop}")
    print(f"WIKI crop path: {args.wiki_crop}")
    print(f"Output path: {args.output}")
    print("=" * 60)
    
    setup_imdb_wiki_structure(args.imdb_crop, args.wiki_crop, args.output)
    
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Update config.age_gender_data_path to point to the output directory")
    print("2. Make sure image paths in label.txt are correct")
    print("3. If images are in different location, update paths in label.txt")

