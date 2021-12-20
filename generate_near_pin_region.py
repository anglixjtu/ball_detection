import cv2
import numpy as np
import matplotlib.pyplot as plt


REF_P2D = [[750, 1082], [1698, 1220], [1874, 1136], [1057, 1005]]
REF_IMG_PATH = 'G:/dataset/golf/sample_video_frames/x10_reference.png'
SAVE_PATH = 'output/golf/x10_reference_near_pin_'
ZOOM_RATE = 10
'''REF_P2D = [[940, 998], [2207, 1198], [1933, 1326], [462, 1110]]
REF_IMG_PATH = 'G:/dataset/golf/sample_video_frames/x13_reference.png'
SAVE_PATH = 'output/golf/x13_reference_near_pin_'
ZOOM_RATE = 13'''
LENS_RANGE = [4.8, 120]
ZOOM_TIMES = 25
CMOS_SIZE = 6.52
NEW_DIST = [1, 1.5, 2, 2.5]


def solve_extrinsic_params(ref_p2d, ref_p3d, camera_matrix):
    """
    Estimate Camera Extrinsic Parameter Matrixes from 2d and 3d correspondence
    """
    dist_coeffs = np.zeros((4, 1))
    imagePoints = np.array(ref_p2d, dtype=np.float32)
    objectPoints = np.array(ref_p3d, dtype=np.float32)

    ret, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints,
                                   camera_matrix, dist_coeffs)
    return ret, rvec, tvec


def estimate_intrinsic_params(zoom_rate, lens_range, zoom_times,
                              cmos_size, PW, PH):
    """
    estimate the intrinsic parameters of the camera.
    """
    # FX = (zoom_rate * (lens_range[1] - lens_range[0]) / zoom_times)
    FX = 60 # 83
    PFX = FX * np.sqrt(PW**2 + PH**2)/cmos_size
    PFY = PFX
    CX = PW / 2.0
    CY = PH / 2.0
    K = np.asarray([[PFX, 0, CX],
                    [0, PFY, CY],
                    [0, 0, 1], ])
    return K, FX


def intersection_point(line1, line2):
    """
    calculate the intersection point of two lines
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    a = (x2 - x1) / (y2 - y1)
    b = (x4 - x3) / (y4 - y3)
    y = (a * y1 - b * y3 + x3 - x1)/(a-b)
    x = a * (y - y1) + x1
    return int(x), int(y)


def draw_regions(regions_pts, img, color=(0, 0, 225), thickness=3):
    for p2d in regions_pts:
        center = p2d[0].astype(np.int32)
        cv2.circle(img, center, radius=10, color=color, thickness=thickness)
    pts = regions_pts.reshape((-1, 1, 2)).astype(np.int32)
    img = cv2.polylines(img, [pts], True, color=color, thickness=thickness)
    return img


def get_new_squares():
    img = cv2.imread(REF_IMG_PATH, cv2.COLOR_BGR2RGB)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    PH, PW, _ = img.shape

    camera_intrinsic, fx = estimate_intrinsic_params(ZOOM_RATE,
                                                     LENS_RANGE,
                                                     ZOOM_TIMES,
                                                     CMOS_SIZE,
                                                     PW, PH)

    # ref_p3d = [[-1000, 0, 1000], [1000, 0, -1000],
    #           [1000, 0, -1000], [-1000, 0, -1000]]
    ref_p3d = [[-1000, 0, 1000], [-1000, 0, -1000],
               [1000, 0, -1000], [1000, 0, 1000]]
    objectPoints = np.array(ref_p3d, dtype=np.float32)

    ret, rvec, tvec = solve_extrinsic_params(
        REF_P2D, ref_p3d, camera_intrinsic)

    objectPointSets = [x * objectPoints for x in NEW_DIST]

    for i, p3d in enumerate(objectPointSets):
        prjt_p2d_list = []
        prjt_p2d = cv2.projectPoints(p3d,  rvec, tvec, camera_intrinsic, None)
        img_show = draw_regions(prjt_p2d[0], img, color=[255, 0, 0])
        for pt in prjt_p2d[0]:
            prjt_p2d_list.append([int(pt[0][0]), int(pt[0][1])])
        print(prjt_p2d_list)
    img_show = draw_regions(np.array([REF_P2D]), img_show, color=[0, 0, 255])
    cv2.imwrite(SAVE_PATH+str(fx)+'.png', img_show)
    plt.imshow(img_show)
    plt.show()


if __name__ == '__main__':
    get_new_squares()
