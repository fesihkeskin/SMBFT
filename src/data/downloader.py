"""Dataset downloader for the SMBFT project.

Downloads Houston13, Pavia_University, and Salinas datasets and
organizes them into the expected folder structure.
"""

# src/data/downloader.py

import os
import requests

# List of dataset URLs
urls = [
    # Salinas
    "https://www.ehu.eus/ccwintco/uploads/f/f1/Salinas.mat",
    "https://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat",
    "https://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat",
       
    # Pavia University
    "https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
    "https://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat",

    # Houston (UH)
    "http://hyperspectral.ee.uh.edu/2egc4t4jd76fd32/Houston.mat",
    "http://hyperspectral.ee.uh.edu/2egc4t4jd76fd32/Houston_gt.mat"
]

# Mapping: dataset → list of expected file names
dataset_folders = {
    "Salinas": [
        "Salinas.mat",
        "Salinas_corrected.mat",
        "Salinas_gt.mat"
    ],
    "Pavia_University": [
        "Pavia_University.mat",
        "Pavia_University_gt.mat"
    ],
    "Houston": [
        "Houston.mat",
        "Houston_gt.mat"
    ]
}

# Original filename to target filename mapping (for normalization)
rename_map = {

    "PaviaU.mat": "Pavia_University.mat",
    "PaviaU_gt.mat": "Pavia_University_gt.mat",
}

def get_dataset_dir() -> str:
    """Return the absolute path to the Dataset directory."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "Dataset")

def download_file(url: str, save_path: str):
    """Download file via HTTP and save to the specified path."""
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"[✅] Downloaded: {os.path.basename(save_path)}")
    except requests.RequestException as e:
        print(f"[❌] Failed to download {url}\n    Reason: {e}")

def prepare_directories(base_dir: str):
    """Create necessary directories for each dataset."""
    for folder in dataset_folders.keys():
        os.makedirs(os.path.join(base_dir, folder), exist_ok=True)

def match_folder(filename: str) -> str:
    """Find the correct folder name given a filename."""
    for folder, files in dataset_folders.items():
        if filename in files:
            return folder
    return None

def download_datasets():
    """Main function to download and organize all dataset files."""
    base_dir = get_dataset_dir()
    prepare_directories(base_dir)

    for url in urls:
        orig_name = os.path.basename(url)
        final_name = rename_map.get(orig_name, orig_name)
        folder = match_folder(final_name)

        if folder:
            save_path = os.path.join(base_dir, folder, final_name)
            if not os.path.exists(save_path):
                download_file(url, save_path)
            else:
                print(f"[=] Already exists: {final_name}")
        else:
            print(f"[!] Warning: No match found for {orig_name}")

    print("\n✅ All downloads complete. ✅")

if __name__ == "__main__":
    download_datasets()