import pydicom as dcm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import skfuzzy as fuzz
from sklearn.metrics import silhouette_score


def jaccard_similarity(cluster1, cluster2):
    intersection = np.logical_and(cluster1, cluster2)
    union = np.logical_or(cluster1, cluster2)
    similarity = np.sum(intersection) / np.sum(union)
    return similarity


def display_results(image, kmeans_labels, fcm_labels, similarity, silhouette_scores):
    # Invert the labels to correct the color mapping
    kmeans_labeled_image = np.max(kmeans_labels) - kmeans_labels
    fcm_labeled_image = np.max(fcm_labels) - fcm_labels

    # Adjust the figure size here to fit your screen
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # Smaller figure size

    # Display the original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Display K-means segmented image
    axes[0, 1].imshow(kmeans_labeled_image, cmap='gray')
    axes[0, 1].set_title('K-means Segmentation')
    axes[0, 1].axis('off')

    # Display Fuzzy C-means segmented image
    axes[1, 0].imshow(fcm_labeled_image, cmap='gray')
    axes[1, 0].set_title('Fuzzy C-means Segmentation')
    axes[1, 0].axis('off')

    text_kmeans = f'Silhouette Score (K-means): {silhouette_scores[0]:.2f}'
    text_fcm = f'Silhouette Score (FCM): {silhouette_scores[1]:.2f}'

    axes[0, 1].text(0.5, -0.1, text_kmeans, ha='center', va='top', transform=axes[0, 1].transAxes, fontsize=10)
    axes[1, 0].text(0.5, -0.1, text_fcm, ha='center', va='top', transform=axes[1, 0].transAxes, fontsize=10)

    # Display Histogram
    axes[1, 1].hist(kmeans_labels.ravel(), bins=50, color='blue', alpha=0.7, label='K-means')
    axes[1, 1].hist(fcm_labels.ravel(), bins=50, color='red', alpha=0.7, label='Fuzzy C-means')
    axes[1, 1].set_title('Histogram of Segmented Intensities')
    axes[1, 1].legend()

    # Add text box for displaying spatial overlap
    textstr = f'Jaccard Similarity: {round(similarity, 3)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axes[1, 1].text(0.05, 0.95, textstr, transform=axes[1, 1].transAxes, fontsize=11,
                    verticalalignment='top', bbox=props)
    plt.tight_layout()  # Optimize the layout
    plt.show()


def match_clusters(kmeans_labels, fcm_labels, image):
    # Calculate the mean intensity for each cluster in K-means
    kmeans_means = [np.mean(image[kmeans_labels == i]) for i in np.unique(kmeans_labels)]

    # Calculate the mean intensity for each cluster in Fuzzy C-means
    fcm_means = [np.mean(image[fcm_labels == i]) for i in np.unique(fcm_labels)]

    # Match clusters based on the closest mean intensity values
    cluster_map = {}

    for kmeans_cluster, kmeans_mean in enumerate(kmeans_means):
        closest_fcm_cluster = np.argmin([abs(kmeans_mean - fcm_mean) for fcm_mean in fcm_means])
        cluster_map[kmeans_cluster] = closest_fcm_cluster

    return cluster_map


def segment_mri_image(image_path, mri_type='T2'):
    ds = dcm.dcmread(image_path)
    # `arr` is a numpy.ndarray
    arr = ds.pixel_array

    img_raw = np.array(arr)
    if img_raw.ndim == 3:
        image = cv2.GaussianBlur(img_raw[75, :, :], (5, 5), 0)
    else:
        image = cv2.GaussianBlur(img_raw, (5, 5), 0)

    # Reshape the image to a 2D array of pixels
    pixels = image.reshape(-1, 1)
    pixels = np.float32(pixels)

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=0).fit(pixels)
    kmeans_labels = kmeans.labels_.reshape(image.shape)

    #Fuzzy
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        pixels.T, 3, 2, error=0.005, maxiter=1000, init=None)
    fcm_labels = np.argmax(u, axis=0).reshape(image.shape)

    tissue_types = ['CSF', 'Gray Matter', 'White Matter']
    intensities = ["Dark", "Medium", "Light"]
    if mri_type == 'T1':
        sorted_kmeans_indices = np.argsort(kmeans.cluster_centers_.flatten())
        sorted_fcm_indices = np.argsort(cntr.flatten())
    elif mri_type == 'T2':
        sorted_kmeans_indices = np.argsort(-kmeans.cluster_centers_.flatten())
        sorted_fcm_indices = np.argsort(-cntr.flatten())
        tissue_types = tissue_types[::-1]
    elif mri_type == 'FLAIR':
        sorted_kmeans_indices = np.argsort(kmeans.cluster_centers_.flatten())
        sorted_fcm_indices = np.argsort(cntr.flatten())
        tissue_types = ['Suppressed CSF', 'Gray Matter', 'White Matter/Lesions']
    else:
        raise ValueError("Invalid MRI type specified. Use 'T1', 'T2', or 'FLAIR'.")

    # Mapping the sorted clusters to specific intensities
    kmeans_labeled_image = np.zeros_like(kmeans_labels)
    fcm_labeled_image = np.zeros_like(fcm_labels)
    for i, (kmeans_idx, fcm_idx) in enumerate(zip(sorted_kmeans_indices, sorted_fcm_indices)):
        kmeans_labeled_image[kmeans_labels == kmeans_idx] = i
        fcm_labeled_image[fcm_labels == fcm_idx] = i

    cluster_map = match_clusters(kmeans_labels, fcm_labels, image)
    # Remapping K-means labels to match Fuzzy C-means labels
    remapped_kmeans_labels = np.vectorize(cluster_map.get)(kmeans_labels)

    # Now calculate the Jaccard Similarity & Silhouette
    similarity = jaccard_similarity(remapped_kmeans_labels, fcm_labels)
    silhouette_kmeans = silhouette_score(pixels, kmeans_labels.flatten())
    silhouette_fcm = silhouette_score(pixels, fcm_labels.flatten())
    display_results(image, remapped_kmeans_labels, fcm_labels, similarity, [silhouette_kmeans, silhouette_fcm])


# Usage
segment_mri_image("MRIs/3222/_AX_FSE_T2/2012-01-24_10_02_48.0/I282797/PPMI_3222_MR__AX_FSE_T2_br_raw_20120206110705289_46_S139717_I282797.dcm", mri_type="T2")
