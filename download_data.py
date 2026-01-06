import os
import shutil
import subprocess

def download_and_setup_data():
    repo_url = "https://github.com/luangtatipsy/intel-image-classification.git"
    temp_dir = "temp_dataset_repo"
    target_dir = "data/intel_image"

    # Clean up temp dir if exists
    if os.path.exists(temp_dir):
        print(f"Cleaning up existing temp dir: {temp_dir}")
        shutil.rmtree(temp_dir)

    # Clone the repository
    print(f"Cloning repository from {repo_url}...")
    try:
        subprocess.check_call(["git", "clone", repo_url, temp_dir])
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone repository: {e}")
        return

    # Define source and destination paths
    # Repo structure:
    # temp_dataset_repo/dataset/seg_train -> train
    # temp_dataset_repo/dataset/seg_test -> val
    # Note: The repo description says "datasets" directory in root, but let's check structure dynamically or assume based on description
    
    repo_data_root = os.path.join(temp_dir, "dataset") # Check if it is 'dataset' or 'datasets'
    if not os.path.exists(repo_data_root):
        repo_data_root = os.path.join(temp_dir, "datasets")
    
    if not os.path.exists(repo_data_root):
        print(f"Could not find dataset directory in {temp_dir}")
        # List dirs to help debug
        print("Dirs in repo:", os.listdir(temp_dir))
        return

    print(f"Found data root at: {repo_data_root}")
    
    # Map repo folders to our structure
    # train -> train
    # test -> val
    
    # Check what's inside repo_data_root
    subdirs = os.listdir(repo_data_root)
    print(f"Subdirectories: {subdirs}")
    
    train_src = None
    val_src = None
    
    if 'seg_train' in subdirs:
        train_src = os.path.join(repo_data_root, 'seg_train')
    if 'seg_test' in subdirs:
        val_src = os.path.join(repo_data_root, 'seg_test')
        
    if not train_src or not val_src:
        print("Could not identify train/test directories.")
        return

    # Move files
    # We need to move content of train_src to target_dir/train
    # And val_src to target_dir/val
    
    # Ensure target dirs exist
    os.makedirs(os.path.join(target_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'val'), exist_ok=True)
    
    print("Moving training data...")
    # Iterate over classes in train_src
    for cls in os.listdir(train_src):
        src_cls_path = os.path.join(train_src, cls)
        dst_cls_path = os.path.join(target_dir, 'train', cls)
        
        if os.path.isdir(src_cls_path):
            if os.path.exists(dst_cls_path):
                shutil.rmtree(dst_cls_path)
            shutil.copytree(src_cls_path, dst_cls_path)
            print(f"  Moved {cls} to train")

    print("Moving validation data...")
    for cls in os.listdir(val_src):
        src_cls_path = os.path.join(val_src, cls)
        dst_cls_path = os.path.join(target_dir, 'val', cls)
        
        if os.path.isdir(src_cls_path):
            if os.path.exists(dst_cls_path):
                shutil.rmtree(dst_cls_path)
            shutil.copytree(src_cls_path, dst_cls_path)
            print(f"  Moved {cls} to val")

    # Cleanup
    print("Cleaning up temp files...")
    # Handle permission errors on windows git folders sometimes
    def on_rm_error(func, path, exc_info):
        import stat
        os.chmod(path, stat.S_IWRITE)
        func(path)
        
    shutil.rmtree(temp_dir, onerror=on_rm_error)
    print("Done!")

if __name__ == "__main__":
    download_and_setup_data()
