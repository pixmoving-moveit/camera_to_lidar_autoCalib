import cv2
import cv2.aruco as aruco
import numpy as np

# 读取图像（请根据实际图片路径修改）
# image = cv2.imread('/home/pix/code/calibration_ws/src/calibBoard/script/1111.png')
image = cv2.imread('/home/pix/code/calibration_ws/src/calibBoard/data/camera_rear_right/image_undistort.png')

# image = cv2.resize(image, (1920, 1080))  # 调整为1080p分辨率，便于显示


if image is None:
    raise FileNotFoundError('未找到图片 rear_right_undistorted.png')


# 曝光增强（简单线性增益，alpha>1增亮）
alpha = 1.5  # 亮度增益
beta = 20    # 亮度偏移

# 保留一份未画坐标轴的原图用于拼接
image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
image_for_patch = image.copy()

#cv2.imwrite('/home/pix/code/calibration_ws/src/calibBoard/data/camera_front/image_undistort.png', image)  # 保存为PNG格式，避免JPEG压缩损失


# 选择 AprilTag 16h5 字典
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_16h5)
parameters = aruco.DetectorParameters_create()
parameters.adaptiveThreshConstant = 1
parameters.errorCorrectionRate = 0.1
parameters.markerBorderBits = 1
parameters.minMarkerPerimeterRate = 0.015   # 默认 0.03，允许更小
parameters.polygonalApproxAccuracyRate = 0.03  # 默认 0.05，更 tolerant


# ====== 显式定义相机内参和畸变系数 ======
camera_matrix = np.array([
    [487.35858154296875, 0.0, 517.1744464523581],
    [0.0, 485.6971435546875, 288.4947253932478],
    [0.0, 0.0, 1.0]
], dtype=np.float64)
dist_coeffs = np.array([
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
], dtype=np.float64)  # 只取前5个参数

# 设置 ArUco 码实际边长（单位：米）
marker_length = 0.51  # 请根据实际情况修改

# 检测 ArUco 标记
corners, ids, rejected = aruco.detectMarkers(image, aruco_dict, parameters=parameters)

# 估算每个 ArUco 的位姿和绘制
output = image.copy()
if ids is not None and len(corners) > 0:
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
    output = aruco.drawDetectedMarkers(output, corners, ids)
    for i, marker_id in enumerate(ids.flatten()):
        print(f"ID={marker_id} rvec={rvecs[i].flatten()} tvec={tvecs[i].flatten()}")
        # 只在output上绘制坐标轴，不影响image_for_patch
        aruco.drawAxis(output, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], marker_length/2)
else:
    print("未检测到 ArUco 标记，无法估算位姿")

# 检测并绘制以右下角为锚点的外围黑色包围框（向左上角扩展）
if corners is not None:

    # 创建同尺寸白底图像
    white_img = np.ones_like(image_for_patch) * 255
    for idx, marker_corners in enumerate(corners):
        pts = marker_corners[0]
        scale = 2.5
        shift_rb = 0.2
        vec_rb2lt = pts[0] - pts[2]
        anchor = pts[2] + vec_rb2lt * shift_rb
        outer_pts = []
        for i in range(4):
            vec = pts[i] - anchor
            outer_pt = anchor + vec * scale
            outer_pts.append(outer_pt)
        outer_pts = np.array(outer_pts, dtype=np.int32)
        cv2.polylines(output, [outer_pts], isClosed=True, color=(0,0,255), thickness=2)
        print(f"外围框顶点: {outer_pts}")

        # 生成掩码并提取ROI
        mask = np.zeros(image_for_patch.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [outer_pts], 255)
        # 只将包围框内像素复制到白底图（用未画坐标轴的image_for_patch）
        white_img[mask == 255] = image_for_patch[mask == 255]

    # 显示白底拼接结果
    cv2.imshow('ROI on White', white_img)




# 在图像上显示相机内参
fx, fy = camera_matrix[0,0], camera_matrix[1,1]
cx, cy = camera_matrix[0,2], camera_matrix[1,2]
dist_str = ', '.join([f'{d:.3f}' for d in dist_coeffs])
text_lines = [
    f"fx={fx:.1f} fy={fy:.1f}",
    f"cx={cx:.1f} cy={cy:.1f}",
    f"dist: [{dist_str}]"
]
for i, line in enumerate(text_lines):
    cv2.putText(output, line, (10, 30+30*i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

# 鼠标点击事件：绘制小红点并输出坐标
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(output, (x, y), 2, (0, 0, 255), -1)
        cv2.imshow('ArUco Detection', output)
        print(f"点击坐标: ({x}, {y})")

cv2.imshow('ArUco Detection', output)
cv2.setMouseCallback('ArUco Detection', mouse_callback)

while True:
    key = cv2.waitKey(1)
    if key == 27:  # 按ESC退出
        break
cv2.destroyAllWindows()
