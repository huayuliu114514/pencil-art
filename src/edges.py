import cv2
import numpy as np

KERNELS = {
    "sharpen1": np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], np.float32),
    "sobelH": np.array([[-1,-2,-1],[0,0,0],[1,2,1]], np.float32),
    "sobelV": np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32),
}

def enhance_edges(img, kernel_name="sharpen1"):
    return cv2.filter2D(img, -1, KERNELS[kernel_name])

def convoluteY(img, weights):
    h, w = img.shape
    side = int(np.sqrt(len(weights)))
    half = side // 2
    out = np.zeros_like(img, dtype=np.float32)
    weights = np.array(weights).reshape(side, side)

    for y in range(h):
        for x in range(w):
            s = 0.0
            for cy in range(side):
                for cx in range(side):
                    iy = min(h-1, max(0, y + cy - half))
                    ix = min(w-1, max(0, x + cx - half))
                    s += img[iy, ix] * weights[cy, cx]
            out[y,x] = s

    return np.clip(out, 0, 255).astype(np.uint8)

def thicken_edges(edge_img, ksize=3, iterations=1):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(edge_img, kernel, iterations)
