import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "four and three leap clovers.jpeg"
image = cv2.imread(image_path)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

clover_path = "four leap clover.jpeg"
clover = cv2.imread(clover_path)
clover_gray = cv2.cvtColor(clover, cv2.COLOR_BGR2GRAY)

w, h = clover_gray.shape[::-1]

res = cv2.matchTemplate(image_gray, clover_gray, cv2.TM_CCOEFF_NORMED)
threshold = 0.87

loc = np.where(res >= threshold)
a = 0
mask = np.zeros_like(res)
for pt in zip(*loc[::-1]):
    if mask[pt[1], pt[0]] == 0: # kareler çakışmasın ve kesişmesin diye maskeleme yaptık
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2) #kare içine alma
        center = (pt[0] + w // 2, pt[1] + h // 2)  # Dikdörtgenin merkez koordinatları
        cv2.circle(image, center, 5, (0, 0, 255), -1)  # Merkezi işaretleme
        x, y = center
        plt.axvline(x=x, color='r', linestyle='--')
        plt.axhline(y=y, color='r', linestyle='--')
        mask[pt[1]:pt[1] + h, pt[0]:pt[0] + w] = 1
        a = a + 1


plt.imshow(image)
plt.axis('on')  # Eksenleri aç

plt.show()


