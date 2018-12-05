import os
from shutil import copy


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


def createNothingClass():
    ##Get names of classes we dont want to take video's from
    badFolderList = []
    nothingClassFolderNames = []

    for filepath, x, x in os.walk((r"data")):
        badFolderList.append(filepath)

    badFolderList.pop(0)

    newList = []
    for item in badFolderList:
        sep = r'data'
        item = item.split(sep, 1)[1]
        item = item[1:]
        print(item)
        newList.append(item)

    badFolderList = newList

    ##Badfolderlist now contains the names of classes we don't want to take videos from

    ##Take some videos from every class, but not from the clasess we are using
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
            ##The number is the ammound of videos taken per class
            if i > 7:
                break
            else:
                copy(filepath, 'data/nothing/')
                i += 1


createNothingClass()
