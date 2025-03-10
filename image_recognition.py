import numpy as np
import cv2 as cv

from sklearn.cluster import DBSCAN
from PIL import Image as Img

import tkinter as tk    
from tkinter.filedialog import askopenfilename

tk.Tk().withdraw() 

def image_recognition(pattern, frames):
    if frames[-3:] == "MOV":
        img1 = cv.imread(pattern)  # queryImage
        img1 = cv.cvtColor(np.array(img1), cv.COLOR_RGB2GRAY)
        vidcap = cv.VideoCapture(frames)

        while True:
            success, img2 = vidcap.read()

            gray_img2 = cv.cvtColor(np.array(img2), cv.COLOR_RGB2GRAY)

            # Initiate SIFT detector
            sift = cv.SIFT_create()

            # Find the key-points and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(gray_img2, None)

            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=70)  # or pass empty dictionary
            flann = cv.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)

            good_matches = []
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good_matches.append(m)

            if len(good_matches) > 50:
                # Extract matched key-points
                matched_points = np.float32([kp2[m.trainIdx].pt for m in good_matches])

                # Applying DBSCAN clustering to find the largest cluster
                epsilon = 80  # Neighborhood radius
                min_samples = 10  # Minimum number of points in the cluster
                dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
                clusters = dbscan.fit_predict(matched_points)

                try:
                    # Removing noise points
                    largest_cluster_label = np.argmax(np.bincount(clusters[clusters != -1]))
                    # Selecting points from the largest cluster
                    largest_cluster_points = matched_points[clusters == largest_cluster_label]

                    # Calculating the coordinates of the rectangle surrounding the points from the largest cluster
                    min_x = np.min(largest_cluster_points[:, 0])
                    max_x = np.max(largest_cluster_points[:, 0])
                    min_y = np.min(largest_cluster_points[:, 1])
                    max_y = np.max(largest_cluster_points[:, 1])

                except:
                    min_x, min_y, max_x, max_y = 0, 0, 0, 0

                # Draw rectangle around the largest cluster points on the original image
                img2_with_rect = cv.rectangle(np.array(img2),
                                              (int(min_x), int(min_y)),
                                              (int(max_x), int(max_y)),
                                              (255, 0, 0),
                                              4)

                img2 = Img.fromarray(img2_with_rect)
                img3 = cv.cvtColor(np.array(img2), cv.IMREAD_COLOR)
                img3 = cv.resize(img3, (0, 0), fx=0.4, fy=0.3)
                cv.imshow("Video Frame", img3)

                # Stop video if 'k' is pressed
                if cv.waitKey(10) & 0xFF == ord('k'):
                    print("Video stopped.")
                    break

        vidcap.release()
        cv.destroyAllWindows()
    
    else:
        img2 = cv.imread(frames) # trainImage
        img1 = cv.imread(pattern) # queryImage
        img1 = cv.cvtColor(np.array(img1), cv.COLOR_RGB2GRAY)

        
        gray_img2 = cv.cvtColor(np.array(img2), cv.COLOR_RGB2GRAY)

        # Initiate SIFT detector
        sift = cv.SIFT_create()

        # Find the key-points and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(gray_img2, None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=70)  # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)


        # Ratio test as per Lowe's paper
        good_matches = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)

        if len(good_matches) > 50:
            # Extract matched key-points
            matched_points = np.float32([kp2[m.trainIdx].pt for m in good_matches])

            # Applying DBSCAN clustering to find the largest cluster
            epsilon = 80  # Neighborhood radius
            min_samples = 10  # Minimum number of points in the cluster
            dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
            clusters = dbscan.fit_predict(matched_points)

            try:
                # Removing noise points
                largest_cluster_label = np.argmax(np.bincount(clusters[clusters != -1]))
                # Selecting points from the largest cluster
                largest_cluster_points = matched_points[clusters == largest_cluster_label]

                # Calculating the coordinates of the rectangle surrounding the points from the largest cluster
                min_x = np.min(largest_cluster_points[:, 0])
                max_x = np.max(largest_cluster_points[:, 0])
                min_y = np.min(largest_cluster_points[:, 1])
                max_y = np.max(largest_cluster_points[:, 1])

            except:
                min_x, min_y, max_x, max_y = 0, 0, 0, 0

            # Draw rectangle around the largest cluster points on the original image
            img2_with_rect = cv.rectangle(np.array(img2),
                                            (int(min_x), int(min_y)),
                                            (int(max_x), int(max_y)),
                                            (255, 0, 0),
                                            4)

            img2 = Img.fromarray(img2_with_rect)
            img3 = cv.cvtColor(np.array(img2), cv.IMREAD_COLOR)
            img3 = cv.resize(img3, (0, 0), fx=0.4, fy=0.3)
            cv.imshow("", img3)
            cv.waitKey(0)


if __name__ == "__main__":
    
    pattern = askopenfilename(title="Choose an pattern image") # show an "Open" dialog box and return the path to the select
    frames = askopenfilename(title="Choose a source image")
    
    image_recognition(pattern, frames)

    


