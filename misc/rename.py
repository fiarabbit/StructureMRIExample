from os import link, listdir
from os.path import join
import re
from random import shuffle

root_dir = "/home/hashimoto/Deep_MRI/COBRE/COBRE"
target_dir = "/home/hashimoto/data"
subject_dirs = sorted(listdir(root_dir))
_ = []

for i, d in enumerate(subject_dirs):
    if re.match("^[0-9]{7}$", d):
        _.append(d)

subject_dirs = _
shuffle(subject_dirs)

for i, d in enumerate(subject_dirs):
    print(d)
    if i < len(subject_dirs) // 4 * 3:
        dest_path = join(target_dir, "train/mprage{}.nii".format(d))
    elif i < len(subject_dirs) // 8 * 7:
        dest_path = join(target_dir, "valid/mprage{}.nii".format(d))
    else:
        dest_path = join(target_dir, "test/mprage{}.nii".format(d))

    src_path = join(root_dir, d, "session_1/anat_1/mprage.nii")
    link(src_path, dest_path)

