import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import spsolve
import argparse

def buid_2D_lap_mat(height, width):
    block_mat = scipy.sparse.lil_matrix((width, width))
    block_mat.setdiag(-1, -1)
    block_mat.setdiag(4)
    block_mat.setdiag(-1, 1)

    lap_mat = scipy.sparse.block_diag([block_mat] * height).tolil()
    lap_mat.setdiag(-1, 1 * width)
    lap_mat.setdiag(-1, -1 * width)

    return lap_mat

def src_center_alignment(src_img, target_img, center):

    height = target_img.shape[0]
    width = target_img.shape[1]

    src_center = (src_img.shape[1]//2, src_img.shape[0]//2)
    align_vec = np.subtract(center, src_center)
    align_y, align_x = align_vec

    affine_trans_mat = np.float32([[1,0,align_y], [0,1,align_x]])
    src_img_aligned = cv2.warpAffine(src_img, affine_trans_mat, (width,height))
    
    return src_img_aligned

def update_lap_mat_to_preserve_edges(lap_mat, centered_mask, width, height):
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if centered_mask[i, j] == 0:
                index = j + i * width
                lap_mat[index, index + width] = 0
                lap_mat[index, index - width] = 0
                lap_mat[index, index] = 1
                lap_mat[index, index + 1] = 0
                lap_mat[index, index - 1] = 0

def linear_equasions_solution(lap_mat, sol_mat, width, height):
    sol = spsolve(lap_mat, sol_mat)
    sol = sol.reshape((height,width))
    sol[255 < sol] = 255
    sol[sol < 0] = 0
    sol = sol.astype('uint8')

    return sol

def poisson_blend(im_src, im_tgt, im_mask, center):
    height = im_tgt.shape[0]
    width = im_tgt.shape[1]
    src_img_aligned = src_center_alignment(im_src,im_tgt,center)
    channels = src_img_aligned.shape[2]
    mask_aligned = src_center_alignment(im_mask,im_tgt,center)
    flat_mask = mask_aligned.flatten()

    lap_mat = buid_2D_lap_mat(height, width)
    comp_lap_mat = lap_mat.tocsc()

    update_lap_mat_to_preserve_edges(lap_mat, mask_aligned, width, height)

    lap_mat = lap_mat.tocsc()
    for color in range(channels):
        flat_src_img = src_img_aligned[0:height, 0:width, color].flatten()
        flat_target_img = im_tgt[0:height, 0:width, color].flatten()

        preserved_mat = comp_lap_mat.dot(flat_src_img)
        preserved_mat[flat_mask==0] = flat_target_img[flat_mask==0]
        sol_mat = preserved_mat
        linear_sol = linear_equasions_solution(lap_mat, sol_mat, width, height)

        im_tgt[0:height,0:width, color] = linear_sol

    im_blend = im_tgt
    return im_blend

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/banana2.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str
                        , default='./data/seg_GT/banana1.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/table.jpg', help='mask file path')
    return parser.parse_args()

if __name__ == "__main__":
    # Load the source and target images
    args = parse()

    im_tgt = cv2.imread(args.tgt_path, cv2.IMREAD_COLOR)
    im_src = cv2.imread(args.src_path, cv2.IMREAD_COLOR)
    if args.mask_path == '':
        im_mask = np.full(im_src.shape, 255, dtype=np.uint8)
    else:
        im_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]

    center = (int(im_tgt.shape[1] / 2), int(im_tgt.shape[0] / 2))

    im_clone = poisson_blend(im_src, im_tgt, im_mask, center)

    cv2.imshow('Cloned image', im_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
