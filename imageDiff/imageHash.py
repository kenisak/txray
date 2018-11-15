from imutils import paths
import argparse
import time
import sys
import cv2
import numpy as np
import os

def dhash(image, hashSize=8):
    # resize the input image, adding a single column (width) so we
    # can compute the horizontal gradient
    resized = cv2.resize(image, (hashSize + 1, hashSize))

    # compute the (relative) horizontal gradient between adjacent
    # column pixels
    diff = np.array(resized[:, 1:] > resized[:, :-1])

    # convert the difference image to a hash
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])



# grab the paths to both the haystack and needle images
print("[INFO] computing hashes for haystack...")
haystackPaths = list(paths.list_images("bilder"))
needlePaths = list(paths.list_images("images"))

# remove the `` character from any filenames containing a space
# (assuming you're executing the code on a Unix machine)
if sys.platform != "win32":
    haystackPaths = [p.replace("\\", "") for p in haystackPaths]
    needlePaths = [p.replace("\\", "") for p in needlePaths]

# grab the base subdirectories for the needle paths, initialize the
# dictionary that will map the image hash to corresponding image,
# hashes, then start the timer
BASE_PATHS = set([p.split(os.path.sep)[-2] for p in needlePaths])
haystack = {}
start = time.time()

#skapa filen
try:
    file = open("dHash_dubbletter.txt", 'w')
except IOError:
    print("Kunde inte skapa filen ", file)

# loop over the haystack paths
for p in haystackPaths:
    # load the image from disk
    image = cv2.imread(p)

    # if the image is None then we could not load it from disk (so
    # skip it)
    if image is None:
        continue

    # convert the image to grayscale and compute the hash
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageHash = dhash(image)

    # update the haystack dictionary
    l = haystack.get(imageHash, [])
    l.append(p)
    haystack[imageHash] = l


# show timing for hashing haystack images, then start computing the
# hashes for needle images
print("[INFO] processed {} images in {:.2f} seconds".format(
    len(haystack), time.time() - start))
print("[INFO] computing hashes for needles...")

# loop over the needle paths
for p in needlePaths:
    # load the image from disk
    image = cv2.imread(p)

    # if the image is None then we could not load it from disk (so
    # skip it)
    if image is None:
        continue

    # convert the image to grayscale and compute the hash
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageHash = dhash(image)
    file.write("path: {}\n".format(p))

    # grab all image paths that match the hash
    matchedPaths = haystack.get(imageHash, [])
    # loop over all matched paths
    for matchedPath in matchedPaths:
        # extract the subdirectory from the image path
        b = p.split(os.path.sep)[-2]
        file.write("matchedPath: {}\n\n".format(matchedPath))

file.close()
        # if the subdirectory exists in the base path for the needle
        # images, remove it
#        if b in BASE_PATHS:
#            BASE_PATHS.remove(b)
#            print("removed", b)

# display directories to check
print("[INFO] check the following directories...")

# loop over each subdirectory and display it
for b in BASE_PATHS:
    print("[INFOrmat] {}".format(b))

