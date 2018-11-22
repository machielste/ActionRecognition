import glob
import os

from shutil import copy2

##Get names of classes we dont want to take video's from
badFolderList = [dI for dI in os.listdir('data') if os.path.isdir(os.path.join('data', dI))]

for pathAndFilename in sorted(glob.iglob(r"F:\backup project\Moments_in_Time_256x256_30fps\training")):
    if pathAndFilename not in badFolderList:
        for videoPathAndFilename in sorted(glob.iglob(pathAndFilename)):
            i = 0
            while i < 20:
                copy2(videoPathAndFilename, "data/nothing/")
                i += 1
