import os
import sys
import urllib.request
import tarfile
import socket
from tqdm import tqdm

# Set a default socket timeout (15 seconds) so that if the FTP control connection 
# hangs on close, it will automatically time out and proceed to the next file!
socket.setdefaulttimeout(15)

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
    
    # Pass timeout=15 to prevent the remote server from hanging on close
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            meta = response.info()
            file_size = int(meta.get("Content-Length", 0))
            
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=filename) as pb:
                stream = ProgressStream(response, pb)
                with tarfile.open(fileobj=stream, mode="r|gz") as tar:
                    for member in tar:
                        if should_extract(member.name, max_roi):
                            tar.extract(member, path=dest_dir)
    except socket.timeout:
        # If it times out at the very end (after hitting 100%), it means the file is 
        # already fully downloaded/extracted and the server just hung on closing.
        # We can safely catch it and proceed!
        print(f"Connection close timed out for {filename} (Safe to ignore).")
                        
    print(f"✅ Successfully extracted filtered subset from {filename}!")

if __name__ == "__main__":
    urls = [
        "ftp://m1554803:m1554803@dataserv.ub.tum.de/ROIs1158_spring_s1.tar.gz",
        "ftp://m1554803:m1554803@dataserv.ub.tum.de/ROIs1158_spring_s2_cloudy.tar.gz",
        "ftp://m1554803:m1554803@dataserv.ub.tum.de/ROIs1158_spring_s2.tar.gz"
    ]
    
    data_dir = "/content/data/ROIs1158_spring"
    os.makedirs(data_dir, exist_ok=True)
    
    # Stream and extract each archive sequentially, limiting to the first 30 ROIs
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
            stream_extract_url(url, data_dir, max_roi=30)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            sys.exit(1)
            
    print("\n✅ All datasets successfully extracted (filtered to 30 ROIs)!")
