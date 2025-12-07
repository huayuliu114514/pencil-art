import cv2
import numpy as np

def generate_shade_map(gray_img, shade_limit=80):
    h, w = gray_img.shape
    shade = np.zeros((h, w, 4), dtype=np.uint8)
    # y = y > shadeLimit ? 0 : 255
    y = np.where(gray_img > shade_limit, 0, 255).astype(np.uint8)
    shade[..., 0] = y
    shade[..., 1] = 128
    shade[..., 2] = 128
    shade[..., 3] = np.random.randint(0, 256, (h, w), dtype=np.uint8)

    return shade
# 新增：下采样 + 上采样（对应 JS: shadeZoom）
def down_up_sample(shade_rgba, zoom=4):
    h, w = shade_rgba.shape[:2]

    small = cv2.resize(shade_rgba, (w // zoom, h // zoom), interpolation=cv2.INTER_AREA)
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    return big
# 新增：把 shadePixel 和 texture 混合
# 对应 JS:
# y = (255-texture)/255 * (y/255) * shadeLight
def merge_shade_texture(shade_rgba, texture_gray, shade_light=40):
    # 只处理 R 通道（shadePixel.data[i]）
    shade_r = shade_rgba[..., 0].astype(np.float32) / 255.0
    tex_f = (255 - texture_gray.astype(np.float32)) / 255.0

    merged = tex_f * shade_r * shade_light
    merged = np.clip(merged, 0, 255).astype(np.uint8)

    return merged
def pencil_shade_pipeline(
        image_path,
        texture_path,
        shade_limit=80,
        shade_zoom=4,
        shade_light=40):

    # 1. 灰度
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 载入纹理
    texture = cv2.imread(texture_path, cv2.IMREAD_GRAYSCALE)
    texture = cv2.resize(texture, (gray.shape[1], gray.shape[0]))

    # 3. 初始 shade (RGBA)
    shade = generate_shade_map(gray, shade_limit=shade_limit)

    # 4. 下采样 + 上采样（模拟 canvasShadeMin）
    shade_processed = down_up_sample(shade, zoom=shade_zoom)

    # 5. 最终融合（只输出 R 通道的最终结果）
    final_shade = merge_shade_texture(
        shade_processed,
        texture,
        shade_light=shade_light
    )

    return gray, final_shade
gray, shade_final = pencil_shade_pipeline("setsuna.png","pencil-texture.jpg")


def clahe_equalization(img, clipLimit=2.0, tileGridSize=(8,8)):
    """自适应直方图均衡化"""
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(img)

gray = clahe_equalization(gray, clipLimit=2.5, tileGridSize=(16,16))
shade_final = clahe_equalization(shade_final, clipLimit=2.0, tileGridSize=(8,8))

cv2.imwrite("2_gray.png", gray)
cv2.imwrite("6_shade_final.png", shade_final)
# gray = cv2.medianBlur(gray, ksize=3)  # 去噪
KERNELS = {
    "sharpen1": np.array([
         [0, -1,  0],
        [-1,  5, -1],
         [0, -1,  0]
    ], dtype=np.float32),

    "sharpen2": np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ], dtype=np.float32),

    "laplace4": np.array([
         [0, -1,  0],
        [-1,  4, -1],
         [0, -1,  0]
    ], dtype=np.float32),

    "laplace8": np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=np.float32),

    "sobelH": np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32),

    "sobelV": np.array([
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1]
    ], dtype=np.float32),
}
def enhance_edges(img, kernel_name="sharpen1"):
    kernel = KERNELS[kernel_name]
    # 使用卷积增强边缘
    enhanced = cv2.filter2D(img, -1, kernel)
    return enhanced
gray = enhance_edges(gray, kernel_name="sharpen1")

def convoluteY(img, weights):
    h, w = img.shape
    side = int(np.sqrt(len(weights)))
    half = side // 2
    out = np.zeros_like(img, dtype=np.float32)
    weights = np.array(weights).reshape((side, side))

    for y in range(h):
        for x in range(w):
            val = 0.0
            for cy in range(side):
                for cx in range(side):
                    iy = min(h-1, max(0, y + cy - half))
                    ix = min(w-1, max(0, x + cx - half))
                    val += img[iy, ix] * weights[cy, cx]
            out[y, x] = val

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out
sobel_x = [
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
]
sobel_y = [
     1,  2,  1,
     0,  0,  0,
    -1, -2, -1
]
# 2. 得到卷积后的边缘图
edge_img_y = convoluteY(gray, sobel_y)
edge_img_x = convoluteY(gray, sobel_x)

edge_img = np.sqrt(edge_img_x.astype(np.float32)**2 + edge_img_y.astype(np.float32)**2)
edge_img = np.clip(edge_img, 0, 255).astype(np.uint8)

h, w = edge_img.shape
rgba_img = np.zeros((h, w, 4), dtype=np.uint8)
rgba_img[..., 0] = edge_img      # R 通道 = 边缘强度
rgba_img[..., 1] = edge_img      # G 通道同 R
rgba_img[..., 2] = edge_img      # B 通道同 R
rgba_img[..., 3] = np.maximum(255 - edge_img, shade_final) 

def thicken_edges(edge_img, ksize=3, iterations=1):
    kernel = np.ones((ksize, ksize), np.uint8)
    dilated = cv2.dilate(edge_img, kernel, iterations=iterations)
    return dilated
#edge_img = thicken_edges(edge_img, ksize=3, iterations=1)
# 4. 保存或显示结果
cv2.imwrite("edge_with_shade.png", rgba_img)
# 对强化后提取的边缘进行去噪
rgba_img = cv2.medianBlur(rgba_img, ksize=5)
cv2.imwrite("edge_with_shade_denoised.png", rgba_img)

# 渐变上色
def apply_gradient_coloring(edge_img, shade_img, gradient_colors=None, bg_color=(255,255,255),edge_strength=2):
    h, w = edge_img.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)
    

    kernel = np.ones((edge_strength, edge_strength), np.uint8)
    edge_img = cv2.dilate(edge_img, kernel, iterations=1)

    # 默认渐变颜色
    if gradient_colors is None:
        gradient_colors = [
            (0.0, (251,186,48)),
            (0.4, (252,114,53)),
            (0.6, (252,53,78)),
            (0.7, (207,54,223)),
            (0.8, (55,181,217)),
            (1.0, (62,182,218))
        ]
    
    # 生成渐变图
    gradient_map = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            t = (x + y) / (w + h)  # 0-1 线性插值
            # 找对应区间
            for i in range(len(gradient_colors)-1):
                if gradient_colors[i][0] <= t <= gradient_colors[i+1][0]:
                    t0, c0 = gradient_colors[i]
                    t1, c1 = gradient_colors[i+1]
                    alpha = (t - t0) / (t1 - t0)
                    r = int(c0[0]*(1-alpha) + c1[0]*alpha)
                    g = int(c0[1]*(1-alpha) + c1[1]*alpha)
                    b = int(c0[2]*(1-alpha) + c1[2]*alpha)
                    gradient_map[y, x] = (r, g, b)
                    break
    
    # 计算透明度 mask
    # 边缘亮度 + 铅笔亮度，取最大值，归一化到 0-1
    alpha_mask = np.maximum(edge_img, shade_img).astype(np.float32) / 255.0
    
    # 输出 = alpha * 渐变 + (1-alpha) * 背景
    output[..., 0] = (alpha_mask * gradient_map[...,0] + (1-alpha_mask)*bg_color[0]).astype(np.uint8)
    output[..., 1] = (alpha_mask * gradient_map[...,1] + (1-alpha_mask)*bg_color[1]).astype(np.uint8)
    output[..., 2] = (alpha_mask * gradient_map[...,2] + (1-alpha_mask)*bg_color[2]).astype(np.uint8)
    
    return output
edge_img = cv2.medianBlur(edge_img, ksize=3)  # 去噪
import bm3d
edge_img = edge_img.astype(np.float32) / 255.0
edge_img = bm3d.bm3d(edge_img, sigma_psd=0.1)*255.0
final_color_img = apply_gradient_coloring(edge_img, shade_final)
cv2.imwrite("final_pencil_color_test.png", final_color_img)



# def merge_shade_edge_colored(shade_final, edge_img, color=(255, 200, 150), edge_strength=2):
#     # --- 1. 边缘增强 ---
#     kernel = np.ones((edge_strength, edge_strength), np.uint8)
#     edge_enhanced = cv2.dilate(edge_img, kernel, iterations=1)

#     # --- 2. 合并 shade 和边缘亮度 ---
#     combined = np.clip(shade_final.astype(np.float32) + edge_enhanced.astype(np.float32), 0, 255).astype(np.uint8)

#     # --- 3. 根据亮度生成彩色 ---
#     h, w = combined.shape
#     colored = np.zeros((h, w, 3), dtype=np.uint8)

#     # 将亮度归一化到 0~1
#     norm = combined.astype(np.float32) / 255.0

#     # 每个通道按亮度比例上色
#     for i in range(3):
#         colored[..., i] = (norm * color[i] + (1 - norm) * 255).astype(np.uint8)  # 黑->颜色, 亮度低->接近白

#     # --- 4. alpha 通道（可选） ---
#     alpha = (combined > 0).astype(np.uint8) * 255
#     rgba = np.dstack([colored, alpha])

#     return rgba, colored

# # === 示例调用 ===
# # shade_final: 灰度图 uint8
# # edge_img: 灰度图 uint8
# rgba_img, bgr_img = merge_shade_edge_colored(shade_final, edge_img, color=(255, 200, 150), edge_strength=2)

# cv2.imwrite("merged_edge_shade_colored.png", rgba_img)   # RGBA
# cv2.imwrite("merged_edge_shade_colored_bgr.png", bgr_img) # BGR

