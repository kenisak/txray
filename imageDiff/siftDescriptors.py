import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('.\\bilder\\sdltbc27f9b.jpg',0)          # queryImage
img2 = cv2.imread('.\\bilder\\sdltbc27f9b_kopia.jpg',0) # trainImage

# Problemet är att det inte finns någon SIFT i cv2, den är patenterad och har lyfts ur.
# Det går att testa SIFT om man laddar hem koden från GITHub och bygger den. Det är för
# testning samt forskningssyfte.
#  Initiate SIFT detector
sift = cv2.SIFT()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2)
plt.imshow(img3),plt.show()