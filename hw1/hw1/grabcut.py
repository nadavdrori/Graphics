import numpy as np
import cv2
import argparse
from sklearn import mixture as mx  # Of the shelf GMM implementation (Stated as allowed).
import math
import igraph as ig   # Of the shelf Graph optimization lib implementation (Stated as allowed).

GC_BGD = 0  # Hard bg pixel
GC_FGD = 1  # Hard fg pixel, will not be used
GC_PR_BGD = 2  # Soft bg pixel
GC_PR_FGD = 3  # Soft fg pixel

PREV_ENERGY = 10**15
THRESHOLD = 1000
PREV_DIFF = 10000
BETA = -1
K = -1
UPDATED_MASK = None

N_LINKS_EDGES = list()
N_LINKS_WEIGHTS = list()

from_index_to_num = dict()
from_num_to_index = dict()


def grabcut(img, rect, n_iter=5):
    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect

    # Convert from absolute cordinates
    w -= x
    h -= y

    #Initalize the inner square to Foreground
    mask[y:y+h, x:x+w] = GC_PR_FGD
    mask[rect[1]+rect[3]//2, rect[0]+rect[2]//2] = GC_FGD

    bgGMM, fgGMM = initalize_GMMs(img, mask)

    num_iters = 7
    for i in range(num_iters):
        # Update GMM
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)

        mask = update_mask(mincut_sets, mask)

        if check_convergence(energy, mincut_sets):
            break

    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM


def initalize_GMMs(img, mask, n_components=5):  # Yotam
    bg_pixels = np.concatenate((img[mask == GC_BGD], img[mask == GC_PR_BGD]))
    fg_pixels = np.concatenate((img[mask == GC_FGD], img[mask == GC_PR_FGD]))

    bgGMM = mx.GaussianMixture(n_components, init_params='kmeans', covariance_type='full').fit(bg_pixels)
    fgGMM = mx.GaussianMixture(n_components, init_params='kmeans', covariance_type='full').fit(fg_pixels)

    width = img.shape[1]
    height = img.shape[0]

    global BETA
    global K
    global UPDATED_MASK
    global N_LINKS_WEIGHTS
    global N_LINKS_EDGES

    N_LINKS_WEIGHTS.clear()
    N_LINKS_EDGES.clear()


    BETA = calc_beta(img)


    for i in range(width):
        for j in range(height):
            if i < width - 1:
                N_LINKS_EDGES.append((j * width + i, j * width + i + 1))
                N_LINKS_WEIGHTS.append(N(img[j][i], img[j][i + 1], (j, i), (j, i + 1)))
            if j < height - 1:
                N_LINKS_EDGES.append((j * width + i, (j + 1) * width + i))
                N_LINKS_WEIGHTS.append(N(img[j][i], img[j + 1][i], (j, i), (j + 1, i)))
                if i < width - 1:
                    N_LINKS_EDGES.append((j * width + i, (j + 1) * width + i + 1))
                    N_LINKS_WEIGHTS.append(N(img[j][i], img[j + 1][i + 1], (j, i), (j + 1, i + 1)))

            from_index_to_num[(j, i)] = j * width + i
            from_num_to_index[j * width + i] = (j, i)

    K = max(N_LINKS_WEIGHTS)
    UPDATED_MASK = mask.copy()

    return bgGMM, fgGMM


def update_GMMs(img, mask, bgGMM:mx.GaussianMixture, fgGMM:mx.GaussianMixture):
    global UPDATED_MASK
    bg_pixels = np.concatenate((img[UPDATED_MASK == GC_BGD], img[UPDATED_MASK == GC_PR_BGD])).reshape(-1, 3)
    fg_pixels = np.concatenate((img[UPDATED_MASK == GC_FGD], img[UPDATED_MASK == GC_PR_FGD])).reshape(-1, 3)

    componentes_bg = [[] for _ in range(bgGMM.n_components)]
    componentes_fg = [[] for _ in range(fgGMM.n_components)]
    predicted_bg = bgGMM.predict(bg_pixels)
    predicted_fg = fgGMM.predict(fg_pixels)

    for i in range(len(bg_pixels)):
        componentes_bg[predicted_bg[i]].append(bg_pixels[i])
    for i in range(len(fg_pixels)):
        componentes_fg[predicted_fg[i]].append(fg_pixels[i])



    for i in range(bgGMM.n_components):
        if len(componentes_bg[i]) > 0:
            cur_bg = np.array(componentes_bg[i])
            mean_bg = np.mean(cur_bg, axis=0)
            cov_matrix_bg = np.add(cv2.calcCovarMatrix(cur_bg, None, cv2.COVAR_NORMAL | cv2.COVAR_ROWS | cv2.COVAR_SCALE)[0], np.identity(3) * 0.001)  # make sure non singular matrix.
            weight_bg = len(cur_bg) / len(bg_pixels)

            bgGMM.weights_[i] = weight_bg
            bgGMM.covariances_[i] = cov_matrix_bg
            bgGMM.means_[i] = mean_bg

        if len(componentes_fg[i]) > 0:
            cur_fg = np.array(componentes_fg[i])
            mean_fg = np.mean(cur_fg, axis=0)
            cov_matrix_fg = np.add(cv2.calcCovarMatrix(cur_fg, None, cv2.COVAR_NORMAL | cv2.COVAR_ROWS| cv2.COVAR_SCALE)[0], np.identity(3) * 0.001)  # make sure non singular matrix.
            weight_fg = len(cur_fg) / len(fg_pixels)

            fgGMM.weights_[i] = weight_fg
            fgGMM.covariances_[i] = cov_matrix_fg
            fgGMM.means_[i] = mean_fg

    return fgGMM, bgGMM


def vectorized_DFor(GMM, pixels):
    total_probs = np.zeros(len(pixels))
    for k in range(GMM.n_components):
        if np.sum(GMM.means_[k]) != 0:
            weight = GMM.weights_[k]
            mean = GMM.means_[k]
            cov = GMM.covariances_[k]
            pi = weight
            sqrt_of_det = math.sqrt(np.linalg.det(cov))
            mu = mean
            inv = np.linalg.inv(cov)
            for i in range(len(pixels)):
                distance_color_mean_vector = np.fabs(np.subtract(pixels[i], mu))
                inner_exponent = -0.5 * np.matmul(np.matmul(np.transpose(distance_color_mean_vector), inv), distance_color_mean_vector)
                total_exponent = math.exp(inner_exponent)
                prob = (pi / sqrt_of_det) * total_exponent
                total_probs[i] += prob


    total_probs[total_probs == 0] = np.exp(-100)
    return -np.log(total_probs)


def calculate_mincut(img, mask, bgGMM, fgGMM):
    global K, UPDATED_MASK, N_LINKS_EDGES, N_LINKS_WEIGHTS
    height, width = img.shape[:2]
    num_pixels = height * width
    FG_S, BG_S = num_pixels, num_pixels + 1

    edges = N_LINKS_EDGES.copy()
    weights = N_LINKS_WEIGHTS.copy()

    unknown_mask = (UPDATED_MASK != GC_BGD) & (UPDATED_MASK != GC_FGD)
    unknown_pixels = img[unknown_mask]

    bg_probs = vectorized_DFor(bgGMM, unknown_pixels)
    fg_probs = vectorized_DFor(fgGMM, unknown_pixels)

    unknown_indices = np.where(unknown_mask.flatten())[0]

    edges.extend([(BG_S, idx) for idx in unknown_indices])
    weights.extend(bg_probs)
    edges.extend([(idx, FG_S) for idx in unknown_indices])
    weights.extend(fg_probs)

    # Add edges for known background and foreground
    bg_indices = np.where(UPDATED_MASK.flatten() == GC_BGD)[0]
    fg_indices = np.where(UPDATED_MASK.flatten() == GC_FGD)[0]

    edges.extend([(BG_S, idx) for idx in bg_indices])
    weights.extend([K] * len(bg_indices))
    edges.extend([(idx, FG_S) for idx in bg_indices])
    weights.extend([0] * len(bg_indices))

    edges.extend([(BG_S, idx) for idx in fg_indices])
    weights.extend([0] * len(fg_indices))
    edges.extend([(idx, FG_S) for idx in fg_indices])
    weights.extend([K] * len(fg_indices))

    G = ig.Graph(n=num_pixels + 2, edges=edges, edge_attrs={'weight': weights}, directed=True)
    cut = G.st_mincut(source=BG_S, target=FG_S, capacity=weights)
    return cut.partition, cut.value


def N(m, n, m_loc, n_loc):
    dist = sum([(m_loc[i] - n_loc[i])**2 for i in range(len(m_loc))]) ** 0.5
    pix_dist = np.linalg.norm(np.array([float(m[i]) - float(n[i]) for i in range(3)]))
    return (50 / dist) * (pow(math.e, - (BETA * pix_dist)))



def calc_beta(img):
    width = img.shape[1]
    height = img.shape[0]

    distance = 0
    num_of_pixels = 0
    for i in range(width):
        for j in range(height):
            if i < width - 1:
                diff = img[j, i] - img[j, i + 1]
                distance += diff.dot(diff)
                num_of_pixels += 1
            if j < height - 1:
                diff = img[j, i] - img[j + 1, i]
                distance += diff.dot(diff)
                num_of_pixels += 1
            if i < width - 1 and j < height - 1:
                diff = img[j, i] - img[j + 1, i + 1]
                distance += diff.dot(diff)
                num_of_pixels += 1
    mean_squared_distance = distance / num_of_pixels
    return 1 / (2 * mean_squared_distance)



def update_mask(mincut_sets, mask):
    global UPDATED_MASK
    width = mask.shape[1]
    height = mask.shape[0]
    bg_pixels, fg_pixels = mincut_sets
    bg_pixels.remove(width * height + 1)
    fg_pixels.remove(width * height)
    for pixel_i in fg_pixels:
        j, i = from_num_to_index[pixel_i]
        if UPDATED_MASK[j][i] == GC_PR_BGD:
            UPDATED_MASK[j][i] = GC_PR_FGD

    for pixel_i in bg_pixels:
        j, i = from_num_to_index[pixel_i]
        if UPDATED_MASK[j][i] == GC_PR_FGD:
            UPDATED_MASK[j][i] = GC_PR_BGD

    for i in range(height):
        for j in range(width):
            if UPDATED_MASK[i][j] == GC_PR_FGD and mask[i][j] != GC_BGD:
                mask[i][j] = GC_FGD
            elif UPDATED_MASK[i][j] == GC_PR_BGD and mask[i][j] != GC_FGD:
                mask[i][j] = GC_BGD

    return mask

def check_convergence(energy, min_cut):
    global PREV_ENERGY
    global PREV_DIFF
    diff = abs(PREV_ENERGY - energy) / PREV_ENERGY
    diff_pix = abs(len(min_cut[0]) - len(min_cut[1]))
    if diff < 10 ** -2 or abs(diff_pix - PREV_DIFF) < THRESHOLD:
        return True
    PREV_ENERGY = energy
    PREV_DIFF = diff_pix
    return False


def cal_metric(predicted_mask, gt_mask):
    correct_pixels = np.sum(predicted_mask == gt_mask)
    total_pixels = gt_mask.size
    accuracy = (correct_pixels / total_pixels)

    intersection = np.logical_and(predicted_mask, gt_mask).sum()
    union = np.logical_or(predicted_mask, gt_mask).sum()
    jaccard_similarity = intersection / union if union > 0 else 0.0

    return accuracy * 100, jaccard_similarity * 100

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='teddy', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()

if __name__ == '__main__':
    # Load an example image and define a bounding box around the object of interest
    args = parse()


    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int,args.rect.split(',')))


    img = cv2.imread(input_path)

    # Run the GrabCut algorithm on the image and bounding box
    mask, bgGMM, fgGMM = grabcut(img, rect)
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]
    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = cal_metric(mask, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])
    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCut Mask', 255 * mask)
    cv2.imshow('GrabCut Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


