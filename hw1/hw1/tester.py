import grabcut as gc
import cv2
import numpy as np
import time 
import os

def cal_metric(predicted_mask, gt_mask):
    # Calculate accuracy
    total_pixels = predicted_mask.size
    correctly_labeled_pixels = np.sum(predicted_mask == gt_mask)
    accuracy = correctly_labeled_pixels / total_pixels

    # Jaccard similarity
    intersection = np.sum((predicted_mask == 1) & (gt_mask == 1))
    union = np.sum((predicted_mask == 1) | (gt_mask == 1))
    jaccard = intersection / union if union != 0 else 0


    return accuracy*100, jaccard*100


def run_grabcut(imgName,rectName,n_iter=5):
    show = True
    save= False

    input_path = f'data/imgs/{imgName}.jpg'
    rect = tuple(map(int, open(f"data/bboxes/{rectName}.txt", "r").read().split(' ')))
    img = cv2.imread(input_path)
    print(rect)

    # # Create a uniform kernel
    # kernel = np.ones((15, 15), np.float32) / (15 * 15)
    #
    # # Apply the blur
    # blurred_image = cv2.filter2D(img, -1, kernel)
    # img = blurred_image

    # Run the GrabCut algorithm on the image and bounding box
    mask, bgGMM, fgGMM = gc.grabcut(img, rect)
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    gt_mask = cv2.imread(f'data/seg_GT/{imgName}.bmp', cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
    acc, jac = cal_metric(mask, gt_mask)

    if show:
        # Apply the final mask to the input image and display the results
        img_cut = img * (mask[:, :, np.newaxis])
        cv2.imshow('Original Image', img)
        cv2.imshow('GrabCut Mask', 255 * mask)
        cv2.imshow('GrabCut Result', img_cut)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save:
        output_dir = f'data/res/final{imgName}.jpg'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        img_cut = img * (mask[:, :, np.newaxis])
        cv2.imwrite(output_dir, img_cut)

    return acc,jac

if __name__ == '__main__':

    images = ["sheep","flower","fullmoon","grave","llama","memorial","stone2","teddy","banana1","banana2","book","bush","fullmoon","cross"]

    for img in images:
        start = time.time()
        print(f'Runing on {img}')
        save = True
        acc, jac = run_grabcut(img,img)
        end = time.time()
        totalTime = (end - start)/60
        print(f'On {img} got: Accuracy={acc}, Jaccard={jac}, took:{totalTime:.2f} mintes')
    
    # img = "banana2"
    # start = time.time()
    # print(f'Runing on {img}')
    # save = True
    # acc, jac = run_grabcut(img,img)
    # end = time.time()
    # totalTime = (end - start)/60
    # print(f'On {img} got: Accuracy={acc}, Jaccard={jac}, took:{totalTime:.2f} mintes')


        
