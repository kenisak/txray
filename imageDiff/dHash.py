from imutils import paths
import time
import sys
import cv2
import numpy as np
import imagehash



def dhash(image, hashSize=8):
    # resize the input image, adding a single column (width) so we
    # can compute the horizontal gradient
    resized = cv2.resize(image, (hashSize + 1, hashSize))

    # compute the (relative) horizontal gradient between adjacent
    # column pixels
    diff = np.array(resized[:, 1:] > resized[:, :-1])
#    cv2.imshow("image", image)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

    # convert the difference image to a hash
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


# grab the paths to both the haystack and needle images
print("[INFO] computing hashes for haystack...")
pathTillBilderna = list(paths.list_images(".\\"))
# nparray = np.array(list(pathTillBilderna), dtype=np.uint8)

# remove the `` character from any filenames containing a space
# (assuming you're executing the code on a Unix machine)
# if sys.platform != "win32":
#    pathTillBilderna = [p.replace("\\", "") for p in pathTillBilderna]
if sys.platform == "win32":
    pathTillBilderna = [p.replace("\ ", " ") for p in pathTillBilderna]

allaBilder = {}
start = time.time()

# skapa filen
try:
    file = open("dHash_dubbletter.txt", 'w')
    fileFel = open("fel.txt", 'w')
    nollVarden = open("nollVarden.txt", 'w')
    hashVarde = open("hashVarde.txt", 'w')
except IOError:
    print("Kunde inte skapa filen {} eller fel filen {}".format(file, fileFel))

# loop för att sätta ett hasvärde på samtliga bilder
for p in pathTillBilderna:
    # Det fanns åäöÅÄÖ i filnamnen fick göra om bilden till en numpyarray
    # och sedan göra en imdecode istället för imread.
    image = 0
    myImageHash = 0
    stream = open(p, "rb")
    bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)
    image = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

    # if the image is None then we could not load it from disk (so
    # skip it)
    if image is None:
        print("Det var nog inte en bild: {}".format(p))
        continue
    try:
        # Konvertera bilden till gråskala och beräkna hash.
        # Lade till denna try, då det visar sig att vi får ett
        # exception av någon anledning, kolla om man kan ta reda
        # på channels innan man konverterar bilden
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        myImageHash = dhash(image)
    except cv2.error as e:
        fileFel.write("Bilden: {}\n Felet: {}\n\n".format(p, e))
        pass

    # Vi kollar ifall det finns bilder som är lika
    l = allaBilder.get(myImageHash, [])
    l.append(p)
    allaBilder[myImageHash] = l
    if myImageHash != 0:
        # skriver alla identiska kopior till filen.
        if len(l)>1:
            file.write("{};{}\n".format(l, myImageHash))
        else:
            # vi sätter ett has-värde på alla bilder som inte är dubletter
            #det kan ju komma in dubletter i framtiden.
            hashVarde.write("{};{}\n".format(l, myImageHash))
    else:
        # Vi får bilder med hash=0, dessa skriv till egen fil.
        # Jag skriver ut sökvägen till varje bild genom att använda
        # pathvariabeln, annars hamnar alla bilder på samma rad.
       nollVarden.write("{}\n".format(p))

print("[INFO] processed {} images in {:.2f} seconds".format(
    len(allaBilder), time.time() - start))



file.close()
fileFel.close()
nollVarden.close()
hashVarde.close()
