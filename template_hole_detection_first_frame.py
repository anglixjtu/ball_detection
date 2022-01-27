import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def parse_args():
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--front_view_path', type=str, dest='front_view_path',
                        default='E:/dataset/golf/reference_frames/reference_frames/first_frames',
                        help='the path to front-view images')
    parser.add_argument('--out_path', type=str, dest='out_path',
                        default='E:/dataset/golf/reference_frames/reference_frames/out_first_frames_front',
                        help='the path to top-view images')

    args = parser.parse_args()
    return args


def top_hole_detection(img, template, mask=None):
    """Detect the hole using template matching
    """

    # All the 6 methods for comparison in a list
    '''methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
               'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']'''
    methods = ['cv2.TM_CCOEFF']
    blur_ksize= 51      #hyper-parameter

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    w, h = template.shape[::-1]

    for meth in methods:
        img_backup = gray.copy()
        method = eval(meth)
        # Apply template Matching
        res = cv2.matchTemplate(gray, template, method)
        h, w = template.shape
        res = cv2.GaussianBlur(res, (blur_ksize, blur_ksize), 0)
        #mask = mask.astype(np.uint8)
        #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res, mask=mask[h//2:-h//2+1, w//2:-w//2+1])
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        matched_region = gray[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]
        '''plt.imshow(matched_region, cmap='gray')
        plt.show()'''

        #  top_left is for a region of temple size
        #  find the darkest point as the center
        cx = top_left[0] + np.argmin(np.mean(matched_region, 0))
        cy = top_left[1] + np.argmin(np.mean(matched_region, 1))
        img_show = cv2.circle(img, (cx, cy), 20, (0,0,255), -1)
        plt.imshow(img)
        plt.show()

        debug =True
    return (cx, cy), img_show


def move_ref_pts(ref_pts, H, W, ox=0, oy=0):
    def line(p1, p2):
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0]*p2[1] - p2[0]*p1[1])
        return A, B, -C

    def intersection(L1, L2):
        D  = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]
        if D != 0:
            x = Dx / D
            y = Dy / D
            return int(x), int(y)
        else:
            return False
    
    L1 = line(ref_pts[0], ref_pts[2])
    L2 = line(ref_pts[1], ref_pts[3])
    ref_center = intersection(L1, L2)
    delta_x = W//2 - ref_center[0]
    delta_y = H//2 - ref_center[1]
    centered_ref_pts = []
    for pt in ref_pts:
        centered_ref_pts.append([int(pt[0])+delta_x-ox,
                                 int(pt[1])+delta_y-oy])
    return np.float32(centered_ref_pts)


def compute_square_size(ref_pts):
    size = 0
    for i in [0, 1, 2, 3]:
        for j in [1, 2, 3, 0]:
            pt1 = ref_pts[i]
            pt2 = ref_pts[j]
            curt_size = np.sqrt((pt1[0]-pt2[0])**2 + 
                                (pt1[1]-pt2[1])**2)
            if curt_size > size:
                size = curt_size
    return int(2*(size//2))


def project_to_top(img, ref_pts, square_size):
    n_warp = 3
    H, W, C = img.shape
    
    half_square_size = square_size // 2

    roi_x0, roi_y0 = 500, 500
    roi_x1, roi_y1 = 2000, 1000
    ref_pts = move_ref_pts(ref_pts, H, W, ox=roi_x0, oy=roi_y0)

    center_h = int(half_square_size*n_warp)
    center_w = int(half_square_size*n_warp)
    pts2 = np.float32([
                       [ center_w-half_square_size, center_h+half_square_size],
                       [ center_w+half_square_size, center_h+half_square_size],
                       [ center_w+half_square_size, center_h-half_square_size],
                       [ center_w-half_square_size, center_h-half_square_size],])

    M = cv2.getPerspectiveTransform(ref_pts, pts2)
    invert_M = cv2.getPerspectiveTransform(pts2, ref_pts)

    dst = cv2.warpPerspective(img[roi_y0:roi_y1, roi_x0:roi_x1], M, (center_w*2, center_h*2),borderValue =(0, 255, 0)) #(center_w*2, center_h*2)
    mask = cv2.warpPerspective(np.ones((roi_y1-roi_y0, roi_x1-roi_x0)), M, (center_w*2, center_h*2))
    
    dst = dst#[h-int(crop_n*bw):h+int(crop_n*bw),
             # w-int(crop_n*bw):w+int(crop_n*bw)]
    '''plt.imshow(dst)
    plt.show()'''
    debug=True
    return dst, mask, invert_M, (roi_x0, roi_y0)


def back_project(hole_pt, M, roi_corner):
    hole_pt = np.array([hole_pt[0], hole_pt[1], 1])
    dst_hole_pt = np.dot(M, hole_pt)
    return (dst_hole_pt[0:2]+roi_corner).astype(np.uint)



def main():
    args = parse_args()
    filenames = os.listdir(args.front_view_path)
    ref_pts = np.float32([[750, 1082], [1698, 1220], [1874, 1136], [1057, 1005]])
    hole = cv2.imread('E:/dataset/golf/reference_frames/reference_frames/top_template.jpg')
    square_size = compute_square_size(ref_pts)
    for filename in filenames:
        filepath = os.path.join(args.front_view_path, filename)
        front_view_img = cv2.imread(filepath)
        top_view_img, mask, invert_M, roi_corner = project_to_top(front_view_img, ref_pts, square_size)
        hole_pt, img_show = top_hole_detection(top_view_img, hole, mask)
        hole_pt_front = back_project(hole_pt, invert_M, roi_corner)
        img_show = cv2.circle(front_view_img, hole_pt_front, 20, (0,0,255), -1)
        plt.imshow(img_show)
        plt.show()
        img_name = os.path.split(filepath)[1]
        out_file = os.path.join(args.out_path, "%s"%img_name)
        if not os.path.exists(args.out_path):
            os.makedirs(args.out_path)
        cv2.imwrite(out_file, img_show)




if __name__ == '__main__':
    # get_new_squares()
    main()