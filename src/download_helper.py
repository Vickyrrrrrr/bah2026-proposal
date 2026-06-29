import os
import sys
import urllib.request
import tarfile
from tqdm import tqdm

class ProgressStream:
    def __init__(self, response, progress_bar):
        self.response = response
        self.pb = progress_bar

    def read(self, amt=None):
        data = self.response.read(amt)
        if data:
            self.pb.update(len(data))
        return data

def stream_extract_url(url, dest_dir):
    filename = os.path.basename(url)
    print(f"\nStreaming & extracting {filename} on-the-fly...")
    
    # Get content length for progress bar
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        meta = response.info()
        file_size = int(meta.get("Content-Length", 0))
        
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=filename) as pb:
            stream = ProgressStream(response, pb)
            with tarfile.open(fileobj=stream, mode="r|gz") as tar:
                tar.extractall(path=dest_dir)
                
    print(f"✅ Successfully extracted {filename}!")

if __name__ == "__main__":
    # URLs for the three spring dataset archives
    urls = [
        "ftp://m1554803:m1554803@dataserv.ub.tum.de/ROIs1158_spring_s1.tar.gz",
        "ftp://m1554803:m1554803@dataserv.ub.tum.de/ROIs1158_spring_s2_cloudy.tar.gz",
        "ftp://m1554803:m1554803@dataserv.ub.tum.de/ROIs1158_spring_s2.tar.gz"
    ]
    
    data_dir = "/content/data/ROIs1158_spring"
    os.makedirs(data_dir, exist_ok=True)
    
    # First, let's clean up any half-downloaded archives in /content/ to free up space immediately
    for filename in ["ROIs1158_spring_s1.tar.gz", "ROIs1158_spring_s2_cloudy.tar.gz", "ROIs1158_spring_s2.tar.gz"]:
        old_file = os.path.join("/content", filename)
        if os.path.exists(old_file):
            try:
                os.remove(old_file)
                print(f"Removed old archive {filename} to free up space.")
            except Exception:
                pass
                
    # Stream and extract each archive sequentially
    for url in urls:
        filename = os.path.basename(url)
        # Check if the folder is already extracted to avoid repeating work
        subfolder_map = {
            "ROIs1158_spring_s1.tar.gz": "s1",
            "ROIs1158_spring_s2_cloudy.tar.gz": "s2_cloudy",
            "ROIs1158_spring_s2.tar.gz": "s2"
        }
        check_folder = os.path.join(data_dir, subfolder_map[filename])
        if os.path.exists(check_folder) and len(os.listdir(check_folder)) > 0:
            print(f"Folder {subfolder_map[filename]} already extracted. Skipping.")
            continue
            
        try:
            stream_extract_url(url, data_dir)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            sys.exit(1)
            
    print("\n✅ All datasets successfully extracted to local disk!")
