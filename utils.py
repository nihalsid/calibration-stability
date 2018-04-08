import xmltodict
import numpy as np
import cv2
import os, sys
import math
from matplotlib import pyplot as plt

config = {
    'grid_bounds': [[-0.5, 0.5], [-0.25, 0.25], [-0.5, 0.5]],
    'grid_widths': [20, 20, 20],
    'max_width': 1944,
    'max_height': 2592,
    'point_in_epiline_fig': 25
}


def generate_grid():
    grid_bounds = config['grid_bounds']
    grid_widths = config['grid_widths']
    indiv_grids = []
    for i, gb in enumerate(grid_bounds):
        indiv_grid = []
        for j in range(grid_widths[i]):
            interval = (grid_bounds[i][1] - grid_bounds[i][0]) / (grid_widths[i] - 1);
            indiv_grid.append(grid_bounds[i][0] + interval * j)
        indiv_grids.append(indiv_grid)
    grid = []
    for i in range(grid_widths[0]):
        for j in range(grid_widths[1]):
            for k in range(grid_widths[2]):
                grid.append(np.array([indiv_grids[0][i], indiv_grids[1][j], indiv_grids[2][k]], dtype=np.float))
    return np.transpose(np.array(grid))


def project_grid_on_image(grid, calibration):
    X = grid
    R = calibration["R"]
    T = calibration["T"]
    K = calibration["K"]
    x = np.matmul(K, np.matmul(R, X) + T)
    for i in range(x.shape[1]):
        if x[2][i] == 0:
            np.delete(x, i, 1)
            np.delete(x, i, 1)
    x = x / x[2, :]
    pts_x = np.transpose(x)[:, 0:2]
    return pts_x


def draw_epilines(lines, pts1, pts2, image1, image2, image_name):
    img2 = image2.copy()
    img1 = image1.copy()
    for l, pt2, pt1 in zip(lines, pts2, pts1):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        if l[1] != 0:
            x0, y0 = map(int, [0, -l[2] / l[1]])
            x1, y1 = map(int, [img2.shape[1], -(l[2] + l[0] * img2.shape[1]) / l[1]])
            img2 = cv2.line(img2, (x0, y0), (x1, y1), color, 3)
            img2 = cv2.circle(img2, tuple(map(int, pt2)), 10, color, -1)
            img1 = cv2.circle(img1, tuple(map(int, pt1)), 10, color, -1)
    cv2.imwrite('epilines/%s' % image_name, np.concatenate((img1, img2), axis=1))


def calculate_error_using_epilines(lines, pts2, pts1):
    average_distance = 0.0
    for l, pt2, pt1 in zip(lines, pts2, pts1):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        # distance of pt2 from line
        a = l[0]
        b = l[1]
        c = l[2]
        dist = np.abs(a * pt2[0] + b * pt2[1] + c) / np.sqrt(a * a + b * b)
        average_distance += dist
    average_distance /= len(lines)
    return average_distance

def get_fundamental_matrix_from_csv(csv1, csv2):
    x1 = np.loadtxt(csv1,delimiter=',')
    x2 = np.loadtxt(csv2,delimiter=',')
    F_im, ransac_mask = cv2.findFundamentalMat(x1, x2, method=cv2.FM_RANSAC, param1=3, param2=0.995)
    return F_im


def get_fundamental_matrix_from_images(image1, image2, save_name):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)
    bf = cv2.BFMatcher()
    # matches = bf.match(des1, des2)
    # good = []
    # for m in matches:
    #     good_match = True
    #     for n in matches:
    #         if m.distance >= 3 * n.distance:
    #             good_match = False
    #     if good_match:
    #         good.append(m)

    matches = bf.knnMatch(des1, des2, k=2)
    matches = sorted(matches, key=lambda x: x[0].distance / x[1].distance)
    good = []
    for m, n in matches:
        if m.distance < 0.70 * n.distance:
            good.append(m)
    # good = [m[0] for m in matches[:50]]

    image_matches = cv2.drawMatches(image1, kp1, image2, kp2, good, flags=2, outImg=None)
    cv2.imwrite('good/%s' % save_name, image_matches)
    num_estimation_points = len(good)
    x1 = np.empty(shape=(num_estimation_points, 2))
    x2 = np.empty(shape=(num_estimation_points, 2))
    for i in range(num_estimation_points):
        x1[i] = kp1[good[i].queryIdx].pt
        x2[i] = kp2[good[i].trainIdx].pt
    F_im, ransac_mask = cv2.findFundamentalMat(x1, x2, method=cv2.FM_RANSAC, param1=3, param2=0.995)
    matches_used_in_ransac = [good[i] for i, rm in enumerate(ransac_mask) if ransac_mask[i] == 1]
    image_matches = cv2.drawMatches(image1, kp1, image2, kp2, matches_used_in_ransac[:8], flags=2, outImg=None)
    cv2.imwrite('sift/%s' % save_name, image_matches)
    return F_im


def get_fundamental_matrix_from_calibrations(calib1, calib2):
    R1 = calib1["R"]
    R2 = calib2["R"]
    T1 = calib1["T"]
    T2 = calib2["T"]
    K1 = calib1["K"]
    K2 = calib2["K"]
    R = np.matmul(R2, np.transpose(R1))
    T = np.subtract(T2, np.matmul(R, T1))
    T_hat = [[0, -T[2], T[1]], [T[2], 0, -T[0]], [-T[1], T[0], 0]]
    F_true = np.matmul(np.matmul(np.transpose(np.linalg.inv(K2)), np.matmul(T_hat, R)), np.linalg.inv(K1))
    return F_true


def read_calibration(calibration_file):
    calibrations = []
    with open(calibration_file, "r") as f:
        parsed_xml = xmltodict.parse(f.read())
    cameras = parsed_xml["calibration"]["camera"]
    for camera in cameras:
        projection = camera["projection"][0]
        K = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float)
        K[0][0] = projection['alpha']
        K[0][1] = projection['skew']
        K[0][2] = projection['principal']['x']
        K[1][0] = 0
        K[1][1] = projection['beta']
        K[1][2] = projection['principal']['y']
        K[2][0] = 0
        K[2][1] = 0
        K[2][2] = 1
        pose = camera["pose"]
        R = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float)
        R[0][0] = pose["rotation"]["matrix"]["a00"]
        R[0][1] = pose["rotation"]["matrix"]["a01"]
        R[0][2] = pose["rotation"]["matrix"]["a02"]
        R[1][0] = pose["rotation"]["matrix"]["a10"]
        R[1][1] = pose["rotation"]["matrix"]["a11"]
        R[1][2] = pose["rotation"]["matrix"]["a12"]
        R[2][0] = pose["rotation"]["matrix"]["a20"]
        R[2][1] = pose["rotation"]["matrix"]["a21"]
        R[2][2] = pose["rotation"]["matrix"]["a22"]
        T = np.array([0, 0, 0], dtype=np.float)
        T[0] = pose["translation"]["x"]
        T[1] = pose["translation"]["y"]
        T[2] = pose["translation"]["z"]
        T = T.reshape((3, 1))
        p_T = np.matmul(-1 * R, T)
        P = {
            "R": R,
            "T": p_T,
            "K": K,
            "WR": np.transpose(R),
            "WT": T
        }
        calibrations.append(P)
    return calibrations


def plot_images(img1, img2):
    plt.subplot(121), plt.imshow(img1)
    plt.subplot(122), plt.imshow(img2)
    plt.show()


def index_not_in_range(pts, range_x, range_y):
    indices = []
    for i, p in enumerate(pts):
        if not ((range_x[0] <= p[0] < range_x[1]) and (range_y[0] <= p[1] < range_y[1])):
            indices.append(i)
    return indices


def filter_grid(grid, calibrations):
    indices_to_remove = []
    for j in range(len(calibrations)):
        pts = project_grid_on_image(grid, calibrations[j])
        not_in_image = index_not_in_range(pts, [0, config['max_width']], [0, config['max_height']])
        indices_to_remove.extend(not_in_image)
    indices_to_remove = sorted(list(set(indices_to_remove)), reverse=True)
    grid = np.delete(grid, indices_to_remove, axis=1)
    return grid


def draw_camera_object(ax, R, T, c):
    base_position = np.array([0, 0, 0], dtype=np.float)
    camera_size = 0.08
    base_back = np.array([0, 0, -camera_size / 2], dtype=np.float)
    upper_left_corner = np.array([-camera_size / 2, -camera_size / 2, 0], dtype=np.float)
    upper_right_corner = np.array([camera_size / 2, -camera_size / 2, 0], dtype=np.float)
    lower_left_corner = np.array([-camera_size / 2, camera_size / 2, 0], dtype=np.float)
    lower_right_corner = np.array([camera_size / 2, camera_size / 2, 0], dtype=np.float)

    all_points = np.array([base_position + upper_left_corner, base_position + upper_right_corner, base_position + lower_right_corner, base_position + lower_left_corner, base_back, base_position], dtype=np.float)
    all_points = np.transpose(np.matmul(R, np.transpose(all_points)) + T)
    ax.plot3D(all_points[[0, 1, 2, 3, 0], 0], all_points[[0, 1, 2, 3, 0], 1], all_points[[0, 1, 2, 3, 0], 2], color=c)
    ax.plot3D(all_points[[0, 4], 0], all_points[[0, 4], 1], all_points[[0, 4], 2], color=c)
    ax.plot3D(all_points[[1, 4], 0], all_points[[1, 4], 1], all_points[[1, 4], 2], color=c)
    ax.plot3D(all_points[[2, 4], 0], all_points[[2, 4], 1], all_points[[2, 4], 2], color=c)
    ax.plot3D(all_points[[3, 4], 0], all_points[[3, 4], 1], all_points[[3, 4], 2], color=c)


def read_images(image_folder):
    images = []
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            images.append(cv2.cvtColor(cv2.imread(os.path.join(image_folder, filename)), cv2.COLOR_BGR2GRAY))
    return images


def draw_points_on_image(pts, image):
    for pt in pts:
        image = cv2.circle(image, tuple(map(int, pt)), 6, (255, 0, 0), -1)
    return image


def draw_points_on_images(grid, calibrations, images):
    for i in range(len(calibrations)):
        pts = project_grid_on_image(grid, calibrations[i])
        img = draw_points_on_image(pts, images[i])
        cv2.imwrite('projected/%d.jpg' % i, img)


# For debugging

def get_fundamental_matrix_from_handpicked_points(image1_idx, image2_idx):
    if image1_idx == 0 and image2_idx == 1:
        x1 = np.array([[222, 1340], [554, 1073], [320, 1875], [533, 690], [360, 1567], [432, 1058], [743, 277], [746, 2049]], dtype=np.float)
        x2 = np.array([[237, 1347], [661, 1061], [344, 1921], [590, 634], [362, 1588], [435, 1048], [783, 256], [752, 2061]], dtype=np.float)
        F_im, ransac_mask = cv2.findFundamentalMat(x1, x2, method=cv2.FM_RANSAC)
        return F_im


def get_angles_from_rotation_matrix(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    print x * 180 / math.pi, y * 180 / math.pi, z * 180 / math.pi
    return np.array([x, y, z])


def xml_rotation_matrix_from_angles(theta):
    theta = np.array(theta, dtype=float)
    theta = theta * math.pi / 180
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.transpose(np.dot(R_z, np.dot(R_y, R_x)))
    xml = '''
    <matrix>
          <a00>%f</a00>
          <a01>%f</a01>
          <a02>%f</a02>
          <a10>%f</a10>
          <a11>%f</a11>
          <a12>%f</a12>
          <a20>%f</a20>
          <a21>%f</a21>
          <a22>%f</a22>
    </matrix>
    ''' % (R[0, 0], R[0, 1], R[0, 2], R[1, 0], R[1, 1], R[1, 2], R[2, 0], R[2, 1], R[2, 2])
    print xml
    return R
