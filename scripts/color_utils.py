# scripts/color_utils.py
"""
Utility functions for extracting dominant colors from images.
"""

import cv2, numpy as np
from sklearn.cluster import KMeans

def get_dominant_colors(image_path, k=3):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(img)
    centers = kmeans.cluster_centers_.astype(int)
    return [tuple(c) for c in centers]

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)
