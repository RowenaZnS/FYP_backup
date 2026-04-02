import torch.utils.data as data
import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import random

try:
    from scipy.io import loadmat
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Cannot read .mat files directly.")

class AgeGenderDataset(torch.utils.data.Dataset):
    def __init__(self, config, dataset=["IMDB", "WIKI", "Adience", "MORPH"], transform=None):
        print(f"AgeGenderDataset.__init__ started, dataset={dataset}", flush=True)
        print(f"  root_path: {config.age_gender_data_path}", flush=True)

        self.root_path = config.age_gender_data_path

        self.image_paths = []
        self.labels = []
        
        self.suffix = ""

        self.sample_num = config.num_image // config.recognition_bz * config.age_gender_bz
        print(f"  sample_num: {self.sample_num:,}", flush=True)

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize([config.img_size, config.img_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        self.weight = dict()
        self.weight["Adience"] = 0.25
        self.weight["MORPH"] = 0.25
        self.weight["IMDB+WIKI"] = 0.5

        flag = True
        self.labels_1 = []
        print("到里这一步")
        while flag:
            print("到里这一步2")
            print(flag)
            if "Adience" in dataset and flag:
                print("到里这一步3")
                # Adience
                self.Adience_path = os.path.join(self.root_path, "Adience")
                self.Adience_data = os.path.join(self.Adience_path, "data")
                self.Adience_label = os.path.join(self.Adience_path, ("label" + self.suffix + ".txt"))

                with open(self.Adience_label, "r") as f:
                    data = f.readlines()

                for i in range(1, len(data)):
                    line = data[i].split()
                    if len(line) > 0:

                        image_path = os.path.join(self.Adience_data, line[0])

                        if not (line[1] == "nan" or line[2] == "nan"):
                            label = [0.0, 0]  # age, gender
                            self.image_paths.append(image_path)
                            label[0] = np.asarray([float(line[1])])

                            if line[2] == "1.0":
                                label[1] = 1

                            self.labels_1.append(label)

                    if len(self.labels_1) >= self.sample_num * self.weight["Adience"]:
                        flag = False
                        break
            else:
                flag = False    
        print("到里这一步4")
        print("  Adience/MORPH section 1 complete", flush=True)
        flag = True
        self.labels_2 = []
        while flag:
            if "MORPH" in dataset and flag:
                print("  Loading MORPH dataset...", flush=True)
                # MORPH
                self.MORPH_path = os.path.join(self.root_path, "MORPH")
                self.MORPH_data = os.path.join(self.MORPH_path, "data")
                self.MORPH_label = os.path.join(self.MORPH_path, ("label" + self.suffix + ".txt"))

                with open(self.MORPH_label, "r") as f:
                    data = f.readlines()

                for i in range(1, len(data)):
                    line = data[i].split()
                    if len(line) > 0:

                        image_path = os.path.join(self.MORPH_data, line[0])

                        if not (line[1] == "nan" or line[2] == "nan"):
                            label = [0.0, 0]  # age, gender
                            self.image_paths.append(image_path)
                            label[0] = np.asarray([float(line[1])])

                            if line[2] == "1.0":
                                label[1] = 1

                            self.labels_2.append(label)

                    if len(self.labels_2) >= self.sample_num * self.weight["MORPH"]:
                        flag = False
                        break
            else:
                # If MORPH not in dataset, break immediately
                flag = False
                break

        self.labels.extend(self.labels_1)
        self.labels.extend(self.labels_2)
        print(f"  After Adience/MORPH: {len(self.labels)} labels", flush=True)

        flag = True
        while flag:

            if "WIKI" in dataset and flag:
                print("  Loading WIKI dataset...", flush=True)
                # WIKI
                self.WIKI_path = os.path.join(self.root_path, "WIKI")
                self.WIKI_data = os.path.join(self.WIKI_path, "data")
                self.WIKI_label = os.path.join(self.WIKI_path, ("label" + self.suffix + ".txt"))
                
                # Check if we have .mat file directly (original format)
                wiki_mat_file = os.path.join(self.root_path, "wiki_crop", "wiki.mat")
                if os.path.exists(wiki_mat_file) and SCIPY_AVAILABLE:
                    # Read from .mat file directly
                    print("Loading WIKI from .mat file...")
                    mat_data = loadmat(wiki_mat_file)
                    
                    dob = mat_data['dob'][0]
                    photo_taken = mat_data['photo_taken'][0]
                    full_path = mat_data['full_path'][0]
                    gender = mat_data['gender'][0]
                    
                    wiki_base_path = os.path.join(self.root_path, "wiki_crop")
                    
                    for i in range(len(dob)):
                        if len(self.labels) >= self.sample_num:
                            flag = False
                            break
                            
                        # Calculate age
                        if not np.isnan(dob[i]) and not np.isnan(photo_taken[i]):
                            birth_year = int(dob[i] / 365.25)
                            photo_year = int(photo_taken[i])
                            age = photo_year - birth_year
                            
                            # Get gender
                            if not np.isnan(gender[i]):
                                gender_val = int(gender[i])
                            else:
                                continue  # Skip if gender is unknown
                            
                            # Get image path
                            img_path = full_path[i][0] if isinstance(full_path[i][0], str) else str(full_path[i][0])
                            full_image_path = os.path.join(wiki_base_path, img_path)
                            
                            # Only add valid entries
                            if age > 0 and os.path.exists(full_image_path):
                                label = [0.0, 0]  # age, gender
                                self.image_paths.append(full_image_path)
                                label[0] = np.asarray([float(age)])
                                if gender_val == 1:
                                    label[1] = 1
                                self.labels.append(label)
                else:
                    # Read from label.txt file (converted format)
                    if os.path.exists(self.WIKI_label):
                        with open(self.WIKI_label, "r") as f:
                            data = f.readlines()

                        for i in range(1, len(data)):
                            line = data[i].split()
                            if len(line) > 0:

                                image_path = os.path.join(self.WIKI_data, line[0])

                                if not (line[1] == "nan" or line[2] == "nan"):
                                    label = [0.0, 0]  # age, gender
                                    self.image_paths.append(image_path)
                                    label[0] = np.asarray([float(line[1])])

                                    if line[2] == "1.0":
                                        label[1] = 1

                                    self.labels.append(label)

                            if len(self.labels) == self.sample_num:
                                flag = False
                                break
                    else:
                        print(f"Warning: WIKI label file not found: {self.WIKI_label}")
                        if not SCIPY_AVAILABLE:
                            print("  Install scipy to read .mat files directly, or run convert_imdb_wiki.py to generate label.txt")

            if "IMDB" in dataset and flag:
                print("  Entering IMDB loading section...", flush=True)
                # IMDB
                self.IMDB_path = os.path.join(self.root_path, "IMDB")
                self.IMDB_data = os.path.join(self.IMDB_path, "data")
                self.IMDB_label = os.path.join(self.IMDB_path, ("label" + self.suffix + ".txt"))
                
                # Check if we have .mat file directly (original format)
                imdb_mat_file = os.path.join(self.root_path, "imdb_crop", "imdb.mat")
                print(f"  Checking for .mat file: {imdb_mat_file}", flush=True)
                print(f"  File exists: {os.path.exists(imdb_mat_file)}, SCIPY_AVAILABLE: {SCIPY_AVAILABLE}", flush=True)
                if os.path.exists(imdb_mat_file) and SCIPY_AVAILABLE:
                    # Read from .mat file directly
                    print("Loading IMDB from .mat file...", flush=True)
                    print("  Step 1: Reading .mat file (this may take a minute)...", flush=True)
                    mat_data = loadmat(imdb_mat_file, simplify_cells=False)
                    print("  Step 2: Extracting data arrays...", flush=True)
                    
                    # Data is nested under 'imdb' key
                    if 'imdb' in mat_data:
                        imdb_data = mat_data['imdb']
                        # Access nested structure: imdb[0,0] contains the struct
                        if isinstance(imdb_data, np.ndarray) and len(imdb_data.shape) >= 2:
                            imdb_struct = imdb_data[0, 0]
                            dob = imdb_struct['dob'][0]
                            photo_taken = imdb_struct['photo_taken'][0]
                            full_path = imdb_struct['full_path'][0]
                            gender = imdb_struct['gender'][0]
                        else:
                            # Try direct access
                            dob = imdb_data['dob'][0]
                            photo_taken = imdb_data['photo_taken'][0]
                            full_path = imdb_data['full_path'][0]
                            gender = imdb_data['gender'][0]
                    elif 'dob' in mat_data:
                        # Direct keys (fallback)
                        dob = mat_data['dob'][0]
                        photo_taken = mat_data['photo_taken'][0]
                        full_path = mat_data['full_path'][0]
                        gender = mat_data['gender'][0]
                    else:
                        available_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                        raise KeyError(f"Could not find 'imdb' or 'dob' key. Available keys: {available_keys}")
                    
                    print(f"  Successfully extracted: {len(dob)} records", flush=True)
                    
                    imdb_base_path = os.path.join(self.root_path, "imdb_crop")
                    total_records = len(dob)
                    print(f"  Step 3: Processing {total_records:,} records, need {self.sample_num:,} samples", flush=True)
                    print("  Step 4: Starting to process records (checking files and calculating age/gender)...", flush=True)
                    
                    valid_count = 0
                    skipped_count = 0
                    for i in range(len(dob)):
                        if len(self.labels) >= self.sample_num:
                            flag = False
                            print(f"  ✓ Reached target: {len(self.labels):,} samples", flush=True)
                            break
                        
                        # Print progress every 50k records (more frequent)
                        if i > 0 and i % 50000 == 0:
                            print(f"  Progress: {i:,}/{total_records:,} records processed, {len(self.labels):,} samples loaded, {valid_count:,} valid, {skipped_count:,} skipped", flush=True)
                            
                        # Calculate age
                        if not np.isnan(dob[i]) and not np.isnan(photo_taken[i]):
                            birth_year = int(dob[i] / 365.25)
                            photo_year = int(photo_taken[i])
                            age = photo_year - birth_year
                            
                            # Get gender
                            if not np.isnan(gender[i]):
                                gender_val = int(gender[i])
                            else:
                                skipped_count += 1
                                continue  # Skip if gender is unknown
                            
                            # Get image path
                            img_path = full_path[i][0] if isinstance(full_path[i][0], str) else str(full_path[i][0])
                            full_image_path = os.path.join(imdb_base_path, img_path)
                            
                            # Only add valid entries
                            if age > 0:
                                # Check file existence (this might be slow)
                                if os.path.exists(full_image_path):
                                    valid_count += 1
                                    label = [0.0, 0]  # age, gender
                                    self.image_paths.append(full_image_path)
                                    label[0] = np.asarray([float(age)])
                                    if gender_val == 1:
                                        label[1] = 1
                                    self.labels.append(label)
                                else:
                                    skipped_count += 1
                            else:
                                skipped_count += 1
                        else:
                            skipped_count += 1
                    
                    # Print final summary and stop loop
                    if len(self.labels) < self.sample_num:
                        print(f"  ⚠️  Warning: Only loaded {len(self.labels):,} samples (need {self.sample_num:,}), data may be insufficient", flush=True)
                    print(f"  ✓ IMDB loading complete: {len(self.labels):,} samples, {valid_count:,} valid, {skipped_count:,} skipped", flush=True)
                    flag = False  # Stop the while loop after IMDB loading
                else:
                    # Read from label.txt file (converted format)
                    if os.path.exists(self.IMDB_label):
                        with open(self.IMDB_label, "r") as f:
                            data = f.readlines()

                        for i in range(1, len(data)):

                            line = data[i].split()
                            if len(line) > 0:

                                image_path = os.path.join(self.IMDB_data, line[0])

                                if not (line[1] == "nan" or line[2] == "nan"):
                                    label = [0.0, 0]  # age, gender
                                    self.image_paths.append(image_path)
                                    label[0] = np.asarray([float(line[1])])

                                    if line[2] == "1.0":
                                        label[1] = 1

                                    self.labels.append(label)

                            if len(self.labels) == self.sample_num:
                                flag = False
                                break
                    else:
                        print(f"Warning: IMDB label file not found: {self.IMDB_label}")
                        if not SCIPY_AVAILABLE:
                            print("  Install scipy to read .mat files directly, or run convert_imdb_wiki.py to generate label.txt")


    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        # Ensure tensor is on CPU for pin_memory compatibility
        if isinstance(img, torch.Tensor):
            img = img.cpu() if img.device.type == 'cuda' else img
        else:
            img = torch.tensor(np.asarray(img), device='cpu')
        label = self.labels[idx]

        return img, label

    def __len__(self):
        return len(self.labels)

class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, config, choose, transform=None):

        self.image_names = []
        self.labels = []
        random.seed(config.seed)

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize([config.img_size, config.img_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        if choose == "train":
            self.CelebA_data = config.CelebA_train_data
            self.CelebA_label = config.CelebA_train_label

            self.sample_num = config.num_image // config.recognition_bz * config.CelebA_bz

            # Check if label file exists, if not try to use CSV format
            if os.path.exists(self.CelebA_label):
                with open(self.CelebA_label, "r") as f:
                    data = f.readlines()

                flag = True
                while flag:
                    for i in range(1, len(data)):
                        r = random.random()
                        if r < 0.5:
                            # Support both space-separated and CSV format
                            line_str = data[i].strip()
                            if ',' in line_str:
                                # CSV format
                                line = line_str.split(',')
                                if len(line) >= 41:  # image_id + 40 attributes
                                    img_name = line[0]
                                    label = [0 for j in range(40)]
                                    for j in range(1, 41):
                                        # Convert -1 to 0, 1 to 1
                                        if line[j].strip() == "1":
                                            label[j-1] = 1
                                    self.image_names.append(img_name)
                                    self.labels.append(label)
                            else:
                                # Space-separated format
                                line = line_str.split()
                                if len(line) >= 41:
                                    self.image_names.append(line[0])
                                    label = [0 for j in range(40)]
                                    for j in range(1, 41):
                                        if line[j] == "1":
                                            label[j-1] = 1
                                    self.labels.append(label)

                            if len(self.labels) == self.sample_num:
                                flag = False
                                break
            else:
                # Try to use CSV file directly
                # CSV files are in the parent directory of image directory
                csv_file = os.path.join(os.path.dirname(os.path.dirname(self.CelebA_data)), "list_attr_celeba.csv")
                partition_file = os.path.join(os.path.dirname(os.path.dirname(self.CelebA_data)), "list_eval_partition.csv")
                
                if os.path.exists(csv_file) and os.path.exists(partition_file):
                    print(f"Loading CelebA from CSV files: {csv_file}")
                    # Load partition info
                    partition_dict = {}
                    with open(partition_file, "r") as f:
                        for line in f.readlines()[1:]:  # Skip header
                            parts = line.strip().split(',')
                            if len(parts) >= 2:
                                partition_dict[parts[0]] = int(parts[1])
                    
                    # Load attributes (0=train, 1=val, 2=test)
                    with open(csv_file, "r") as f:
                        lines = f.readlines()
                        header = lines[0].strip().split(',')
                        total_lines = len(lines)
                        print(f"Loading CelebA from CSV: {total_lines:,} lines, need {self.sample_num:,} samples", flush=True)
                        
                        flag = True
                        iteration = 0
                        while flag:
                            iteration += 1
                            for i in range(1, len(lines)):
                                r = random.random()
                                if r < 0.5:
                                    line = lines[i].strip().split(',')
                                    if len(line) >= 41:
                                        img_name = line[0]
                                        # Only use training set (partition == 0)
                                        if partition_dict.get(img_name, -1) == 0:
                                            label = [0 for j in range(40)]
                                            for j in range(1, 41):
                                                # Convert -1 to 0, 1 to 1
                                                if line[j].strip() == "1":
                                                    label[j-1] = 1
                                            self.image_names.append(img_name)
                                            self.labels.append(label)
                                            
                                            # Print progress every 10k samples
                                            if len(self.labels) % 10000 == 0:
                                                print(f"CelebA loading: {len(self.labels):,}/{self.sample_num:,} samples (iteration {iteration})", flush=True)
                                            
                                            if len(self.labels) == self.sample_num:
                                                flag = False
                                                break
                else:
                    raise FileNotFoundError(f"CelebA label file not found: {self.CelebA_label}, and CSV files not found either")
        elif choose == "val":
            self.CelebA_data = config.CelebA_val_data
            self.CelebA_label = config.CelebA_val_label

            if os.path.exists(self.CelebA_label):
                with open(self.CelebA_label, "r") as f:
                    data = f.readlines()

                for i in range(1, len(data)):
                    line_str = data[i].strip()
                    if ',' in line_str:
                        # CSV format
                        line = line_str.split(',')
                        if len(line) >= 41:
                            img_name = line[0]
                            label = [0 for j in range(40)]
                            for j in range(1, 41):
                                if line[j].strip() == "1":
                                    label[j - 1] = 1
                            self.image_names.append(img_name)
                            self.labels.append(label)
                    else:
                        # Space-separated format
                        line = line_str.split()
                        if len(line) >= 41:
                            self.image_names.append(line[0])
                            label = [0 for j in range(40)]
                            for j in range(1, 41):
                                if line[j] == "1":
                                    label[j - 1] = 1
                            self.labels.append(label)
            else:
                # Try to use CSV file directly
                # CSV files are in the parent directory of image directory
                csv_file = os.path.join(os.path.dirname(os.path.dirname(self.CelebA_data)), "list_attr_celeba.csv")
                partition_file = os.path.join(os.path.dirname(os.path.dirname(self.CelebA_data)), "list_eval_partition.csv")
                
                if os.path.exists(csv_file) and os.path.exists(partition_file):
                    print(f"Loading CelebA validation from CSV files")
                    # Load partition info
                    partition_dict = {}
                    with open(partition_file, "r") as f:
                        for line in f.readlines()[1:]:  # Skip header
                            parts = line.strip().split(',')
                            if len(parts) >= 2:
                                partition_dict[parts[0]] = int(parts[1])
                    
                    # Load attributes (1=val)
                    with open(csv_file, "r") as f:
                        for line in f.readlines()[1:]:  # Skip header
                            parts = line.strip().split(',')
                            if len(parts) >= 41:
                                img_name = parts[0]
                                # Only use validation set (partition == 1)
                                if partition_dict.get(img_name, -1) == 1:
                                    label = [0 for j in range(40)]
                                    for j in range(1, 41):
                                        if parts[j].strip() == "1":
                                            label[j - 1] = 1
                                    self.image_names.append(img_name)
                                    self.labels.append(label)
                else:
                    raise FileNotFoundError(f"CelebA label file not found: {self.CelebA_label}, and CSV files not found either")

        elif choose == "test":
            self.CelebA_data = config.CelebA_test_data
            self.CelebA_label = config.CelebA_test_label

            if os.path.exists(self.CelebA_label):
                with open(self.CelebA_label, "r") as f:
                    data = f.readlines()

                for i in range(1, len(data)):
                    line_str = data[i].strip()
                    if ',' in line_str:
                        # CSV format
                        line = line_str.split(',')
                        if len(line) >= 41:
                            img_name = line[0]
                            label = [0 for j in range(40)]
                            for j in range(1, 41):
                                if line[j].strip() == "1":
                                    label[j - 1] = 1
                            self.image_names.append(img_name)
                            self.labels.append(label)
                    else:
                        # Space-separated format
                        line = line_str.split()
                        if len(line) >= 41:
                            self.image_names.append(line[0])
                            label = [0 for j in range(40)]
                            for j in range(1, 41):
                                if line[j] == "1":
                                    label[j - 1] = 1
                            self.labels.append(label)
            else:
                # Try to use CSV file directly
                # CSV files are in the parent directory of image directory
                csv_file = os.path.join(os.path.dirname(os.path.dirname(self.CelebA_data)), "list_attr_celeba.csv")
                partition_file = os.path.join(os.path.dirname(os.path.dirname(self.CelebA_data)), "list_eval_partition.csv")
                
                if os.path.exists(csv_file) and os.path.exists(partition_file):
                    print(f"Loading CelebA test from CSV files")
                    # Load partition info
                    partition_dict = {}
                    with open(partition_file, "r") as f:
                        for line in f.readlines()[1:]:  # Skip header
                            parts = line.strip().split(',')
                            if len(parts) >= 2:
                                partition_dict[parts[0]] = int(parts[1])
                    
                    # Load attributes (2=test)
                    with open(csv_file, "r") as f:
                        for line in f.readlines()[1:]:  # Skip header
                            parts = line.strip().split(',')
                            if len(parts) >= 41:
                                img_name = parts[0]
                                # Only use test set (partition == 2)
                                if partition_dict.get(img_name, -1) == 2:
                                    label = [0 for j in range(40)]
                                    for j in range(1, 41):
                                        if parts[j].strip() == "1":
                                            label[j - 1] = 1
                                    self.image_names.append(img_name)
                                    self.labels.append(label)
                else:
                    raise FileNotFoundError(f"CelebA label file not found: {self.CelebA_test_label}, and CSV files not found either")

    def __getitem__(self, idx):

        img_path = os.path.join(self.CelebA_data, self.image_names[idx])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        # Ensure tensor is on CPU for pin_memory compatibility
        if isinstance(img, torch.Tensor):
            img = img.cpu() if img.device.type == 'cuda' else img
        else:
            img = torch.tensor(np.asarray(img), device='cpu')
        label = self.labels[idx]

        return img, label

    def __len__(self):
        return len(self.image_names)

class ExpressionDataset(torch.utils.data.Dataset):
    def __init__(self, config, transform=None):

        self.image_paths = []
        self.labels = []

        # self.RAF_data = config.RAF_data
        # self.RAF_label = config.RAF_label
        self.AffectNet_data = config.AffectNet_data
        self.AffectNet_label = config.AffectNet_label

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize([config.img_size, config.img_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        self.sample_num = config.num_image // config.recognition_bz * config.expression_bz

        # self.weight = dict()
        # self.weight["RAF"] = 0.5
        # self.weight["AffectNet"] = 0.5

        with open(self.AffectNet_label, "r") as f:
            data = f.readlines()

        # Map emotion names to class IDs (7 classes: 0-6)
        emotion_to_id = {
            'neutral': 0, 'happiness': 1, 'happy': 1, 'surprise': 2, 
            'sadness': 3, 'sad': 3, 'anger': 4, 'disgust': 5, 
            'fear': 6
            # Note: contempt is excluded as model expects 7 classes (0-6)
        }

        total_lines = len(data)
        print(f"Loading AffectNet: {total_lines:,} lines, need {self.sample_num:,} samples", flush=True)

        flag = True
        iteration = 0
        while flag:
            iteration += 1
            for i in range(1, len(data)):
                # Support both CSV format (comma-separated) and space-separated format
                line = data[i].strip()
                if ',' in line:
                    # CSV format: index,path,emotion_name,score
                    parts = line.split(',')
                    if len(parts) >= 3:
                        image_path = parts[1]  # path column
                        emotion_name = parts[2].lower()  # emotion name
                        emotion_id = emotion_to_id.get(emotion_name, -1)
                        if emotion_id >= 0 and emotion_id < 7:  # Only add valid emotions (0-6)
                            full_image_path = os.path.join(self.AffectNet_data, image_path)
                            # Only add if file exists
                            if os.path.exists(full_image_path):
                                self.image_paths.append(full_image_path)
                                self.labels.append(emotion_id)
                else:
                    # Original space-separated format
                    line_parts = line.split()
                    if len(line_parts) >= 3:
                        image_path = os.path.join(self.AffectNet_data, line_parts[0])
                        label_id = int(line_parts[2])
                        # Only add if file exists and label is valid (0-6)
                        if os.path.exists(image_path) and 0 <= label_id < 7:
                            self.image_paths.append(image_path)
                            self.labels.append(label_id)
                
                # Print progress every 10k samples
                if len(self.labels) > 0 and len(self.labels) % 10000 == 0:
                    print(f"AffectNet loading: {len(self.labels):,}/{self.sample_num:,} samples (iteration {iteration}, line {i:,}/{total_lines:,})", flush=True)
                
                # Check if we have enough samples
                if len(self.labels) >= self.sample_num:
                    flag = False
                    break

        # with open(self.RAF_label, "r") as f:
        #     data = f.readlines()

        # flag = True
        # while flag:
        #     for i in range(0, len(data)):
        #         line = data[i].strip('\n').split(" ")

        #         image_name = line[0]
        #         sample_temp = image_name.split("_")[0]

        #         if sample_temp == "train":
        #             image_path = os.path.join(self.RAF_data, image_name)
        #             self.image_paths.append(image_path)
        #             self.labels.append(int(line[1]) - 1)

        #         if len(self.labels) == self.sample_num:
                
                if len(self.labels) >= self.sample_num:
                    flag = False
                    break

        self.labels = np.asarray(self.labels)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        if not os.path.exists(img_path):
            # Return a black image if file doesn't exist
            img = Image.new('RGB', (112, 112), (0, 0, 0))
        else:
            img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        # Ensure tensor is on CPU for pin_memory compatibility
        if isinstance(img, torch.Tensor):
            img = img.cpu() if img.device.type == 'cuda' else img
        else:
            img = torch.tensor(np.asarray(img), device='cpu')
        label = self.labels[idx]

        return img, label

    def __len__(self):
        return len(self.labels)

class RAFDataset(torch.utils.data.Dataset):
    def __init__(self, config, choose, transform=None):

        self.image_names = []
        self.labels = []

        self.RAF_data = config.RAF_data
        self.RAF_label = config.RAF_label

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize([config.img_size, config.img_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        self.train = True if choose == "train" else False

        if choose == "train":
            self.sample_num = config.num_image // config.recognition_bz * config.RAF_bz

            with open(self.RAF_label, "r") as f:
                data = f.readlines()

            flag = True
            while flag:
                for i in range(0, len(data)):
                    line = data[i].strip('\n').split(" ")

                    image_name = line[0]
                    sample_temp = image_name.split("_")[0]

                    if self.train and sample_temp == "train":
                        self.image_names.append(image_name)
                        self.labels.append(int(line[1]) - 1)

                    if len(self.labels) == self.sample_num:
                        flag = False
                        break

        elif choose == "test":

            with open(self.RAF_label, "r") as f:
                data = f.readlines()

            for i in range(0, len(data)):
                line = data[i].strip('\n').split(" ")

                image_name = line[0]
                sample_temp = image_name.split("_")[0]

                if not self.train and sample_temp == "test":
                    self.image_names.append(image_name)
                    self.labels.append(int(line[1]) - 1)

        self.labels = np.asarray(self.labels)

    def __getitem__(self, idx):

        img_path = os.path.join(self.RAF_data, self.image_names[idx])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        # Ensure tensor is on CPU for pin_memory compatibility
        if isinstance(img, torch.Tensor):
            img = img.cpu() if img.device.type == 'cuda' else img
        else:
            img = torch.tensor(np.asarray(img), device='cpu')
        label = torch.tensor(self.labels[idx], device='cpu')

        return img, label

    def __len__(self):
        return len(self.image_names)


class FGnetDataset(torch.utils.data.Dataset):
    def __init__(self, config, choose="all", id=0, transform=None):

        self.FGnet_data = config.FGnet_data
        self.FGnet_label = config.FGnet_label
        self.leave_out_file_name = ""

        self.image_names = []
        self.labels = []

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize([config.img_size, config.img_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        with open(self.FGnet_label, "r") as f:
            data = f.readlines()

        cut = len(data) // 10 * 9
        if choose == "remove_one":
            for i in range(1, len(data)):
                line = data[i].split()
                if i == id:
                    self.leave_out_file = os.path.join(self.FGnet_data, line[0])
                else:
                    self.image_names.append(line[0])
                    label = np.asarray([float(line[1])])
                    self.labels.append(label)
        elif choose == "all":
            for i in range(1, len(data)):
                line = data[i].split()
                self.image_names.append(line[0])
                label = np.asarray([float(line[1])])
                self.labels.append(label)
        elif choose == "9_fold":
            for i in range(1, cut):
                line = data[i].split()
                self.image_names.append(line[0])
                label = np.asarray([float(line[1])])
                self.labels.append(label)
        elif choose == "1_fold":
            for i in range(cut, len(data)):
                line = data[i].split()
                self.image_names.append(line[0])
                label = np.asarray([float(line[1])])
                self.labels.append(label)

    def __getitem__(self, idx):

        img_path = os.path.join(self.FGnet_data, self.image_names[idx])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        # Ensure tensor is on CPU for pin_memory compatibility
        if isinstance(img, torch.Tensor):
            img = img.cpu() if img.device.type == 'cuda' else img
        else:
            img = torch.tensor(np.asarray(img), device='cpu')
        label = torch.tensor(self.labels[idx], device='cpu')

        return img, label

    def __len__(self):
        return len(self.image_names)

    def get_leave_out_file_name(self):
        return self.leave_out_file_name
        
class LAPDataset(torch.utils.data.Dataset):
    def __init__(self, config, choose="", transform=None):

        if choose == "train":
            self.LAP_data = config.LAP_train_data
            self.LAP_label = config.LAP_train_label
        elif choose == "test":
            self.LAP_data = config.LAP_test_data
            self.LAP_label = config.LAP_test_label

        self.image_names = []
        self.labels = []

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize([config.img_size, config.img_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])


        with open(self.LAP_label, "r") as f:
            data = f.readlines()
        

        for i in range(0, len(data)):
            line = data[i].split(";")
    
            self.image_names.append(line[0])
    
            label = [np.asarray([float(line[1])]), np.asarray([float(line[2])])]
    
            self.labels.append(label)

    def __getitem__(self, idx):

        img_path = os.path.join(self.LAP_data, self.image_names[idx])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        # Ensure tensor is on CPU for pin_memory compatibility
        if isinstance(img, torch.Tensor):
            img = img.cpu() if img.device.type == 'cuda' else img
        else:
            img = torch.tensor(np.asarray(img), device='cpu')
        label = self.labels[idx]

        return img, label

    def __len__(self):
        return len(self.image_names)
