import os
import argparse

# Default file extensions to check for being unused
DEFAULT_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.svg', '.tif', '.tiff', '.csv', '.npz', '.json', '.zip', '.tar.gz', '.hdf5', '.h5', '.DMP'}

# File extensions to search IN for references
SEARCH_EXTS = {'.ipynb', '.py', '.md'}

# Folders to completely ignore
DEFAULT_IGNORE_DIRS = {'.git', '.venv', '__pycache__', '.pytest_cache', '.vscode', '.ipynb_checkpoints', '05-AdvancedSegmentation/images', '05-AdvancedSegmentation/annotations', '06-Shapes/input'}

def get_all_candidate_files(extensions, ignore_dirs):
    candidates = []
    for root, dirs, files in os.walk('.'):
        # Prune ignored dirs
        dirs[:] = [d for d in dirs if d not in ignore_dirs and os.path.join(root, d).lstrip('./').rstrip('/') not in ignore_dirs]
        
        for file in files:
            path = os.path.join(root, file)
            # Check for multiple dots (e.g. .tar.gz)
            for ext in extensions:
                if path.lower().endswith(ext):
                    candidates.append(path)
                    break
    return candidates

def get_all_code_content():
    all_content = ""
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in DEFAULT_IGNORE_DIRS]
        for file in files:
            if os.path.splitext(file)[1].lower() in SEARCH_EXTS:
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        all_content += f.read() + "\n"
                except Exception as e:
                    print(f"Error reading {path}: {e}")
    return all_content

def main():
    parser = argparse.ArgumentParser(description="Find potentially unused data/image files in the project.")
    parser.add_argument("--extensions", nargs='+', default=list(DEFAULT_EXTENSIONS), help="File extensions to check.")
    parser.add_argument("--ignore", nargs='+', default=list(DEFAULT_IGNORE_DIRS), help="Directories to ignore.")
    parser.add_argument("--delete", action="store_true", help="Delete identified unused files.")
    
    args = parser.parse_args()

    extensions = {ext.lower() if ext.startswith('.') else '.' + ext.lower() for ext in args.extensions}
    ignore_dirs = set(args.ignore)

    print("Collating code content for references...")
    code_content = get_all_code_content()
    
    print("Finding candidate files...")
    candidates = get_all_candidate_files(extensions, ignore_dirs)
    
    unused = []
    for path in candidates:
        basename = os.path.basename(path)
        # Search for basename in code content
        if basename not in code_content:
            unused.append(path)
    
    print(f"\nFound {len(unused)} potentially unused files out of {len(candidates)} candidates.\n")
    
    for u in sorted(unused):
        if args.delete:
            print(f"Deleting {u}...")
            try:
                os.remove(u)
            except Exception as e:
                print(f"Error deleting {u}: {e}")
        else:
            print(u)

if __name__ == "__main__":
    main()
