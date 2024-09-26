import cv2

def get_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Left button of the mouse is clicked - position (x: ", x, ", y: ", y, ")")

def display_image_with_click(image):
    # 获取图片的宽度和高度
    height, width = image.shape[:2]

    # 设置窗口大小为图片宽度和高度的一半
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    # 五层框架的原图要缩小，钢尺的不用
    cv2.resizeWindow("image", width // 2, height // 2)  # 五层框架
    # cv2.resizeWindow("image", width, height)  # 钢尺

    cv2.setMouseCallback('image', get_mouse_click)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 示例用法
# 假设 image 是你想要显示的图片
image = cv2.imread(r'E:\pythonProject\DeepLearning\RealFlow\5.30_experiment\original_dataset\five_floors_frameworks\0-4_0%\0.png')
display_image_with_click(image)
