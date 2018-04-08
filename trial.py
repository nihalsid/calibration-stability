import numpy as np
import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread('templeSparseRing/templeSR0001.png')
img2 = cv2.imread('templeSparseRing/templeSR0002.png')

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
'''
img3 = None
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[19:20], flags=2, outImg=img3)
plt.imshow(img3), plt.show()
'''

NUM_EST_POINTS = len(matches)

x1 = np.empty(shape=(NUM_EST_POINTS, 2))
x2 = np.empty(shape=(NUM_EST_POINTS, 2))

for i in range(NUM_EST_POINTS):
    x1[i] = kp1[matches[i].queryIdx].pt
    x2[i] = kp2[matches[i].trainIdx].pt


K1 = [[1520.400000, 0.000000, 302.320000], [0.000000, 1525.900000, 246.870000], [0.000000, 0.000000, 1.000000]]
K2 = K1

F_im, _ = cv2.findFundamentalMat(x1, x2)

R1 = [[0.02187598221295043000, 0.98329680886213122000, -0.18068986436368856000],
      [0.99856708067455469000, -0.01266114646423925600, 0.05199500709979997700],
      [0.04883878372068499500, -0.18156839221560722000, -0.98216479887691122000]]
T1 = [-0.0726637729648, 0.0223360353405, 0.614604845959]

R2 = [[-0.03472199972816788400, 0.98429285136236500000, -0.17309524976677537000],
      [0.93942192751145170000, -0.02695166652093134900, -0.34170169707277304000],
      [-0.34099974317519038000, -0.17447403941185566000, -0.92373047190496216000]]
T2 = [-0.0746307029819, 0.0338148092011, 0.600850565131]

R = np.matmul(R2, np.transpose(R1))
T = np.subtract(T2, np.matmul(R, T1))

T_hat = [[0, -T[2], T[1]], [T[2], 0, -T[0]], [-T[1], T[0], 0]]

F_true = np.matmul(np.matmul(np.transpose(np.linalg.inv(K2)), np.matmul(T_hat, R)), np.linalg.inv(K1))

# print F_true
for i in range(NUM_EST_POINTS):
    X2 = [[x2[i][0]], [x2[i][1]], [1]]
    X1 = [[x1[i][0]], [x1[i][1]], [1]]
    # print np.matmul(np.matmul(np.transpose(X2), F_im),X1)

def drawLines(img2, img1, lines, pts2, pts1):
    _, c, _ = img2.shape
    average_distance = 0.0
    for l, pt2, pt1 in zip(lines, pts2, pts1):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -l[2] / l[1]])
        x1, y1 = map(int, [c, -(l[2] + l[0] * c) / l[1]])
        print x0,y0,' -- ',x1,y1
        img2 = cv2.line(img2, (x0, y0), (x1, y1), color, 1)
        img2 = cv2.circle(img2, tuple(map(int, pt2)), 3, color, -1)
        img1 = cv2.circle(img1, tuple(map(int, pt1)), 3, color, -1)
        # distance of pt2 from line
        a = l[0]/l[1]
        b = 1
        c = l[2]/l[1]
        dist = np.abs(a*pt2[0]+b*pt2[1]+c)/np.sqrt(a*a+b*b)
        average_distance += dist
    average_distance /= len(lines)
    return img2, img1, average_distance


i = 0
#pts1 = np.array([x1[-1]])
#pts2 = np.array([x2[-1]])
pts1 = x1[0:8]
pts2 = x2[0:8]
lines12 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F_im)
lines12 = lines12.reshape(-1, 3)
img2_c, img1_c, avg_err = drawLines(img2, img1, lines12, pts2, pts1)

print avg_err

plt.subplot(121), plt.imshow(img1_c)
plt.subplot(122), plt.imshow(img2_c)
plt.show()
