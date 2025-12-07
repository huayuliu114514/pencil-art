import cv2
import numpy as np

def generate_shade_map(gray_img, shade_limit=80):
    h, w = gray_img.shape
    shade = np.zeros((h, w, 4), dtype=np.uint8)
    y = np.where(gray_img > shade_limit, 0, 255).astype(np.uint8)
    shade[..., 0] = y
    shade[..., 1] = 128
    shade[..., 2] = 128
    shade[..., 3] = np.random.randint(0, 256, (h, w), dtype=np.uint8)
    return shade

def down_up_sample(shade_rgba, zoom=4):
    h, w = shade_rgba.shape[:2]
    small = cv2.resize(shade_rgba, (w // zoom, h // zoom), interpolation=cv2.INTER_AREA)
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return big

def merge_shade_texture(shade_rgba, texture_gray, shade_light=40):
    shade_r = shade_rgba[..., 0].astype(np.float32) / 255.0
    tex_f = (255 - texture_gray.astype(np.float32)) / 255.0
    merged = tex_f * shade_r * shade_light
    return np.clip(merged, 0, 255).astype(np.uint8)
