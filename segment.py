import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pydicom as dcm


color = ["Black", "Dark Gray", "Light Gray", 'White']


def segment(pixels, mri_type):
    # Preprocessing: Gaussian blur
    image = cv2.GaussianBlur(pixels, (5, 5), 0)

    # Reshape the image to a 2D array of pixels
    pixels = image.reshape(-1, 1)
    pixels = np.float32(pixels)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=0).fit(pixels)
    labels = kmeans.labels_.reshape(image.shape)
    # Sort the cluster centers and set tissue types based on MRI type
    if mri_type == 'T1':
        sorted_cluster_indices = np.argsort(kmeans.cluster_centers_.flatten())
        tissue_types = ['CSF', 'Gray Matter', 'White Matter']
    elif mri_type == 'T2':
        sorted_cluster_indices = np.argsort(-kmeans.cluster_centers_.flatten())
        tissue_types = ['White Matter', 'Gray Matter', 'CSF']
    elif mri_type == 'FLAIR':
        sorted_cluster_indices = np.argsort(kmeans.cluster_centers_.flatten())
        tissue_types = ['Suppressed CSF', 'Gray Matter', 'White Matter/Lesions']
    else:
        raise ValueError("Invalid MRI type specified. Use 'T1', 'T2', or 'FLAIR'.")

    # Map the sorted clusters to specific intensities
    labeled_image = np.zeros_like(labels)
    for i, cluster_index in enumerate(sorted_cluster_indices):
        labeled_image[labels == cluster_index] = i
    return labeled_image


def segment_mri_image(image_path, mri_type='T1'):
    # Load the image
    ds = dcm.dcmread(image_path)
    # `arr` is a numpy.ndarray
    arr = ds.pixel_array

    img_raw = np.array(arr)

    # Preprocessing: Gaussian blur
    image = cv2.GaussianBlur(img_raw, (5, 5), 0)

    # Reshape the image to a 2D array of pixels
    pixels = image.reshape(-1, 1)
    pixels = np.float32(pixels)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=0).fit(pixels)
    labels = kmeans.labels_.reshape(image.shape)

    # Sort the cluster centers and set tissue types based on MRI type
    if mri_type == 'T1':
        sorted_cluster_indices = np.argsort(kmeans.cluster_centers_.flatten())
        tissue_types = ['Air', 'CSF', 'Gray Matter', 'White Matter']
    elif mri_type == 'T2':
        sorted_cluster_indices = np.argsort(-kmeans.cluster_centers_.flatten())
        tissue_types = ['Air', 'White Matter', 'Gray Matter', 'CSF']
    elif mri_type == 'FLAIR':
        sorted_cluster_indices = np.argsort(kmeans.cluster_centers_.flatten())
        tissue_types = ['Air', 'Suppressed CSF', 'Gray Matter', 'White Matter/Lesions']
    else:
        raise ValueError("Invalid MRI type specified. Use 'T1', 'T2', or 'FLAIR'.")

    # Map the sorted clusters to specific intensities
    labeled_image = np.zeros_like(labels)
    for i, cluster_index in enumerate(sorted_cluster_indices):
        labeled_image[labels == cluster_index] = i

    # Visualization
    fig = plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(labeled_image, cmap='gray')
    plt.title('K-means Segmented Image')
    label_y_position = 0.105  # Adjust this value as needed to position the labels
    for i, tissue in enumerate(tissue_types):
        fig.text(0.5, label_y_position - 0.030 * i, f'{tissue}: {color[i]}', ha='center', fontsize=12)
    plt.axis('off')

    plt.show()
    return labeled_image


# Usage
segment_mri_image(
    'MRIs/101039/PPMI_101039_MR_3D_T1__br_raw_20210924155017056_64_S1066201_I1496277.dcm', mri_type='T1')  # Replace with actual path and specify MRI type
