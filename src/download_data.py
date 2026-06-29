import os
import subprocess
from pathlib import Path

def download_dataset_subset(dest_dir, split="test", season="spring"):
    """
    Downloads a small subset of the SEN12MS-CR dataset using rsync.
    - split="test" (m1659251, password: m1659251) ~15 GB total
    - split="train" (m1554803, password: m1554803) ~300 GB total
    """
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    if split == "test":
        rsync_url = "rsync://m1659251@dataserv.ub.tum.de/m1659251"
        password = "m1659251"
    else:
        rsync_url = "rsync://m1554803@dataserv.ub.tum.de/m1554803"
        password = "m1554803"
        
    # We restrict the download to a single season/ROI folder to conserve local disk space
    # Example folder: ROIs1158_spring or ROIs1868_summer
    folder_prefix = "ROIs1158_spring" if season == "spring" else "ROIs1868_summer"
    
    cmd = [
        "rsync", "-chavzP",
        f"{rsync_url}/{folder_prefix}/",
        str(dest_path / folder_prefix)
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    print(f"Rsync password required: {password}")
    
    # Run the command
    # Note: On Windows, rsync is typically available if git-bash, MSYS2, or WSL is installed.
    # If rsync is not on path, we fall back to printing the instructions.
    try:
        # We pass password using environmental variable RSYNC_PASSWORD
        env = os.environ.copy()
        env["RSYNC_PASSWORD"] = password
        
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream the output
        for line in process.stdout:
            print(line, end="")
            
        process.wait()
        if process.returncode == 0:
            print(f"\n[SUCCESS] Successfully downloaded {folder_prefix} to {dest_dir}")
        else:
            print(f"\n[ERROR] Rsync exited with return code {process.returncode}")
            
    except FileNotFoundError:
        print("\n[WARNING] 'rsync' command not found on this Windows machine.")
        print("To download the dataset, please install rsync (e.g. via Git Bash or WSL) or run this command in WSL / Git Bash:")
        print(f"  rsync -chavzP {rsync_url}/{folder_prefix}/ {dest_dir}/{folder_prefix}")
        print(f"  Password: {password}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download SEN12MS-CR dataset subset")
    parser.add_argument("--dest", type=str, default="./data", help="Destination folder")
    parser.add_argument("--split", type=str, choices=["train", "test"], default="test", help="Dataset split")
    parser.add_argument("--season", type=str, choices=["spring", "summer"], default="spring", help="Season subset")
    args = parser.parse_args()
    
    download_dataset_subset(args.dest, args.split, args.season)
