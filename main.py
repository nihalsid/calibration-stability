from utils import config, read_calibration, generate_grid, project_grid_on_image, get_fundamental_matrix_from_calibrations, calculate_error_using_epilines, filter_grid, draw_camera_object, read_images, draw_points_on_images, get_fundamental_matrix_from_images, get_angles_from_rotation_matrix, xml_rotation_matrix_from_angles, draw_epilines, \
    get_fundamental_matrix_from_handpicked_points, get_fundamental_matrix_from_csv
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys, os
import numpy as np
import argparse

if not (os.path.exists('projected')):
    os.mkdir('projected')
if not (os.path.exists('sift')):
    os.mkdir('sift')
if not (os.path.exists('epilines')):
    os.mkdir('epilines')
if not (os.path.exists('good')):
    os.mkdir('good')


parser = argparse.ArgumentParser()
requiredArgs = parser.add_argument_group('required arguments')
requiredArgs.add_argument("-i", "--input", help="Folder containing an images/ folder and Calibration.xml", required=True)
parser.add_argument("--rerun", action='store_true', help="Use this flag if this is the second run of the script, first run stores the point grid visible which is used from that point on")

args = parser.parse_args()
calibrations = read_calibration(os.path.join(args.input,"Calibration.xml"))
images = read_images(os.path.join(args.input,"images") )
if not args.rerun:
    grid = generate_grid()
    grid = filter_grid(grid, calibrations)
    np.save('grid.npy',grid)
else:
    grid = np.load('grid.npy')
draw_points_on_images(grid, calibrations, images)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['#808080', '#000000', '#FF0000', '#800000', '#808000', '#008000', '#00FFFF', '#008080', '#800080']
for i, calibration in enumerate(calibrations):
    draw_camera_object(ax, calibration["WR"], calibration["WT"], colors[i%len(colors)])

ax.scatter(grid[0, :], grid[1, :], grid[2, :])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([-0.4, 0.4])
ax.set_ylim([-0.6, 0.2])
ax.set_zlim([-0.4, 0.4])

# original [-90.393871567,-0.554226480989,80.3167290335]
# get_angles_from_rotation_matrix(calibrations[0]["WR"])
# xml_rotation_matrix_from_angles([-120.393871567,-0.554226480989,80.3167290335])
# sys.exit(0)

plt.show()


print "%10s\t| %20s\t| %20s" % ("Cameras", "Error(calibration)", "Error(SIFT)")
print 60 * '-'

images = read_images(os.path.join(args.input,"images") )

pts_idx_permuted = np.random.permutation(grid.shape[1])

# pts1 = project_grid_on_image(grid, calibrations[0])
# pts2 = project_grid_on_image(grid, calibrations[1])
# lines12_using_handpicked = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, get_fundamental_matrix_from_handpicked_points(0,1))
# lines12_using_handpicked = lines12_using_handpicked.reshape(-1, 3)
# avg_err_using_handpicked = calculate_error_using_epilines(lines12_using_handpicked, pts2, pts1)
# draw_epilines(lines12_using_handpicked, [pts1[i] for i in pts_idx_permuted[:config['point_in_epiline_fig']]], [pts2[i] for i in pts_idx_permuted[:config['point_in_epiline_fig']]], images[0], images[1], "handpicked0.jpg")
# print "%10s\t| %20.3e\t| %20.2f" % ("[H] %d vs %d" % (0, 1), 0, avg_err_using_handpicked)

for j in range(1, len(calibrations)):
    pts1 = project_grid_on_image(grid, calibrations[j - 1])
    pts2 = project_grid_on_image(grid, calibrations[j])
    lines12_using_calib = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, get_fundamental_matrix_from_calibrations(calibrations[j - 1], calibrations[j]))
    lines12_using_calib = lines12_using_calib.reshape(-1, 3)
    avg_err_using_calib = calculate_error_using_epilines(lines12_using_calib, pts2, pts1)
    lines12_using_images = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, get_fundamental_matrix_from_images(images[j - 1], images[j], "%d.jpg"%(j-1)))
    #lines12_using_images = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, get_fundamental_matrix_from_csv("matches/matches1_%d.csv"%(j), "matches/matches2_%d.csv"%(j)))
    lines12_using_images = lines12_using_images.reshape(-1, 3)
    avg_err_using_images = calculate_error_using_epilines(lines12_using_images, pts2, pts1)
    draw_epilines(lines12_using_images, [pts1[i] for i in pts_idx_permuted[:config['point_in_epiline_fig']]], [pts2[i] for i in pts_idx_permuted[:config['point_in_epiline_fig']]], images[j-1], images[j], "%d.jpg"%(j-1))
    print "%10s\t| %20.3e\t| %20.2f" % ("%d vs %d" % (j - 1, j), avg_err_using_calib, avg_err_using_images)
