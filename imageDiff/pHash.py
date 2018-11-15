from PIL import Image
from imutils import paths
import imagehash
import argparse
import shelve
import glob
import time
import sys


pathTillBilderna = list(paths.list_images(".\\bilder"))
allaBilder = {}
if sys.platform == "win32":
    pathTillBilderna = [p.replace("\ ", " ") for p in pathTillBilderna]

# skapa filen
try:
    file = open("dHash.txt", 'w')
    fileFel = open("fel2.txt", 'w')
    nollVarden = open("nollVarden2.txt", 'w')
    hashVarde = open("hashVarde2.txt", 'w')
    dubblett = open("dHash_dubbletter.txt", 'w')
except IOError:
    print("Kunde inte skapa filen {} eller fel filen {}".format(file, fileFel))
start = time.time()
i = 0

for p in pathTillBilderna:
    image = Image.open(p)
    h = str(imagehash.dhash(image))
    hashen = imagehash.dhash(image)

    filename = p[p.rfind("/") + 1:]
    file.write("{};{}\n".format(p, h))

    l = allaBilder.get(hashen, [])
    l.append(p)
    allaBilder[hashen] = l

    # skriver alla identiska kopior till filen.
    if len(l) > 1:
        dubblett.write("{};{}\n".format(l, hashen))
    else:
        # vi sätter ett has-värde på alla bilder som inte är dubletter
        # det kan ju komma in dubletter i framtiden.
        hashVarde.write("{};{}\n".format(l, hashen))
    i = i + 1

print("[INFO] processed {} images allaBilder.len(): {} och in {:.2f} seconds".format(
    i, len(allaBilder), time.time() - start))

# for a in allaBilder:
#    print("{}".format(allaBilder[a]))


fileFel.close()
file.close()
nollVarden.close()
hashVarde.close()


