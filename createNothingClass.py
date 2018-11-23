import os

##Get names of classes we dont want to take video's from
badFolderList = []
nothingClassFolderNames = []

for filepath, x, x in os.walk((r"data")):
    badFolderList.append(filepath)

badFolderList.pop(0)

for item in badFolderList:
    # print(item)
    sep = r'data'
    item = item.split(sep, 1)[1]
    item = item[1:]
    print(item)

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
    for filepath, x, x in os.walk((r"F:\backup project\Moments_in_Time_256x256_30fps\training")):
        if i > 20: break
        # copy here
        i += 1

print('\n'.join('{}: {}'.format(*k) for k in enumerate(nothingClassFolderNames)))
