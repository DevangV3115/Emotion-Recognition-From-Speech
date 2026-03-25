import os
import urllib.request
import zipfile

# URL for RAVDESS audio-only speech dataset
DATASET_URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip"
ZIP_FILE_PATH = "Audio_Speech_Actors_01-24.zip"
EXTRACT_DIR = "data"

def download_dataset():
    if not os.path.exists(EXTRACT_DIR):
        os.makedirs(EXTRACT_DIR)

    # Check if we already have the zip
    if not os.path.exists(ZIP_FILE_PATH):
        print(f"Downloading dataset from {DATASET_URL}...")
        try:
            # Adding a User-Agent header can sometimes help with downloads
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            
            urllib.request.urlretrieve(DATASET_URL, ZIP_FILE_PATH)
            print("Download completed.")
        except Exception as e:
            print(f"Failed to download dataset. Error: {e}")
            return
    else:
        print("Zip file already exists. Skipping download.")

    # Extract the zip file
    print("Extracting zip file...")
    try:
        with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        print("Extraction completed.")
    except Exception as e:
        print(f"Failed to extract zip file. Error: {e}")

if __name__ == "__main__":
    download_dataset()
