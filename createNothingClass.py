import os
from shutil import copy


##This script is a little messy, but it works perfectly...

def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


##Get names of classes we dont want to take video's from
badFolderList = []
nothingClassFolderNames = []

for filepath, x, x in os.walk((r"data")):
    badFolderList.append(filepath)

badFolderList.pop(0)

newList = []
for item in badFolderList:
    # print(item)
    sep = r'data'
    item = item.split(sep, 1)[1]
    item = item[1:]
    print(item)
    newList.append(item)

badFolderList = newList


##Badfolderlist now contains the names of classes we don't want to take videos from



for filepath, y, g in os.walk((r"F:\backup project\Moments_in_Time_256x256_30fps\training")):

    if any(x in filepath for x in badFolderList):
        print("Bad class found")
        continue
    else:
        nothingClassFolderNames.append(filepath)

nothingClassFolderNames.pop(0)

for item in nothingClassFolderNames:
    i = 0
    for filepath in absoluteFilePaths(item):
        print(filepath)
        if i > 7:
            break
        else:
            copy(filepath, 'data/nothing/')
            i += 1
