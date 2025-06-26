#this function copies files from one location to another based on a heuristic in the file name

import os
import shutil

def copy_filtered_pngs(source_dir: str, destination_dir: str) -> None:
    """
    Copies all .png files from source_dir to destination_dir if the filename contains a specific flag.
    
    Parameters:
        source_dir (str): The root directory to search for .png files.
        destination_dir (str): The directory where matching files will be copied.
        flag (str): Substring that must be present in the filename to qualify for copying.
    """
    if not os.path.isdir(source_dir):
        raise NotADirectoryError(f"Source directory does not exist: {source_dir}")
    
    os.makedirs(destination_dir, exist_ok=True)
    
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(".png"):
                end = os.path.basename(file).split('_')
                V = end[4].split('V')[1]
                V = float(V)
                Vflag = V>800 and V<4000
                f = float(end[2].split('F')[1])
                f = f*0.5 - 0.5
                f_flag = f>5 and f < 80
                if Vflag and f_flag:
                    source_file = os.path.join(root, file)
                    relative_path = os.path.relpath(root, source_dir)
                    target_folder = os.path.join(destination_dir, relative_path)
                    os.makedirs(target_folder, exist_ok=True)
                    dest_file = os.path.join(target_folder, file)
                    shutil.copy2(source_file, dest_file)
                    print(f"Copied: {source_file} -> {dest_file}")


src = 'D:\\DAS\\FK\\meta_halfday_newspeed20250621T123640'
dest = 'D:\\DAS\\FK\\speedsubset'

copy_filtered_pngs(src,dest)