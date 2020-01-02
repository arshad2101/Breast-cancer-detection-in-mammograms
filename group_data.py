import os
import shutil
import errno

# path to all images, normal and abnormal scan
all_dataset_path = 'database/'
path_to_normal_dataset = 'dataset/normal/'
path_to_abnormal_dataset = 'dataset/abnormal/'
path_to_mias = "data_info/mammogram_info.csv"
strings = ("CALC", "CIRC", "SPIC", "MISC", "ARCH", "ASYM")

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

make_sure_path_exists(path_to_normal_dataset)
make_sure_path_exists(path_to_abnormal_dataset)

print("[Begin Processing]")
with open(path_to_mias, 'r') as f:
    for line in f:
        if "NORM" in line:
            src = all_dataset_path + str(line.split(' ')[0]) + '.jpg' # source files
            dst = path_to_normal_dataset + str(line.split(' ')[0]) + '.jpg' # destination to normal files
            shutil.move(src, dst) # copying from database to normal folder

        if any(s in line for s in strings):
            src1 = all_dataset_path + str(line.split(' ')[0]) + '.jpg' # source files
            dst1 = path_to_abnormal_dataset + str(line.split(' ')[0]) + '.jpg' # destination to abnormal files
            shutil.move(src1, dst1) # copying from database to abnormal folder

print('[Finished processing!]')