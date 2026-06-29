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

def should_extract(member_name, max_roi=30):
    """
    Filters files so we only extract the first N Regions of Interest (ROIs).
    This keeps disk usage extremely low (~15 GB instead of 70 GB) and speeds up training.
    """
    parts = member_name.replace('\\', '/').split('/')
    for part in parts:
        if '_' in part:
            subparts = part.split('_')
            if len(subparts) >= 2:
                last = subparts[-1]
                if last.isdigit():
                    roi_id = int(last)
                    if roi_id > max_roi:
                        return False
    return True

def stream_extract_url(url, dest_dir, max_roi=30):
    filename = os.path.basename(url)
    print(f"\nStreaming & extracting first {max_roi} ROIs from {filename} on-the-fly...")
    
    # Get content length for progress bar
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        meta = response.info()
        file_size = int(meta.get("Content-Length", 0))
        
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=filename) as pb:
            stream = ProgressStream(response, pb)
            with tarfile.open(fileobj=stream, mode="r|gz") as tar:
                # Iterate and extract only the first N ROIs
                for member in tar:
                    if should_extract(member.name, max_roi):
                        tar.extract(member, path=dest_dir)
                        
    print(f"✅ Successfully extracted filtered subset from {filename}!")

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
                
    # Stream and extract each archive sequentially, limiting to the first 30 ROIs
    # 30 ROIs = ~750 high-quality patches, which is plenty for training and fits in ~15 GB storage!
    for url in urls:
        filename = os.path.basename(url)
        try:
            stream_extract_url(url, data_dir, max_roi=30)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            sys.exit(1)
            
    print("\n✅ All datasets successfully extracted (filtered to 30 ROIs)!")
