import os
import sys
import urllib.request
from tqdm import tqdm
import tarfile

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(url)) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def extract_tar(archive_path, dest_dir):
    print(f"Extracting {os.path.basename(archive_path)} to {dest_dir}...")
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(path=dest_dir)
    print("Extraction complete!")

if __name__ == "__main__":
    # URLs for the three spring dataset archives
    urls = [
        "ftp://m1554803:m1554803@dataserv.ub.tum.de/ROIs1158_spring_s1.tar.gz",
        "ftp://m1554803:m1554803@dataserv.ub.tum.de/ROIs1158_spring_s2_cloudy.tar.gz",
        "ftp://m1554803:m1554803@dataserv.ub.tum.de/ROIs1158_spring_s2.tar.gz"
    ]
    
    local_dir = "/content"
    data_dir = "/content/data/ROIs1158_spring"
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Download and extract each archive sequentially
    for url in urls:
        filename = os.path.basename(url)
        local_path = os.path.join(local_dir, filename)
        
        # 1. Download if not already fully downloaded
        if os.path.exists(local_path):
            print(f"File {filename} already exists locally. Skipping download.")
        else:
            print(f"\nDownloading {filename}...")
            try:
                download_url(url, local_path)
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                sys.exit(1)
        
        # 2. Extract
        try:
            extract_tar(local_path, data_dir)
        except Exception as e:
            print(f"Error extracting {filename}: {e}")
            sys.exit(1)
            
        # 3. Clean up the archive to free disk space immediately
        try:
            os.remove(local_path)
            print(f"Removed temporary archive {filename}")
        except Exception as e:
            print(f"Error removing archive: {e}")
            
    print("\n✅ All datasets downloaded, extracted, and cleaned up successfully!")
