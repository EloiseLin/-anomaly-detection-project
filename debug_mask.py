
import cv2
import numpy as np

# =============================================
# 只需改这一行：填入你任意一张原始图片的完整路径
TEST_IMAGE = r"D:\raw_data\0503.jpg"
# =============================================

img = cv2.imread(TEST_IMAGE)
if img is None:
    print("❌ 图片读取失败！请检查路径是否正确")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---------- 显示原图 ----------
cv2.imshow("1. 原图 (灰度)", gray)

# ---------- 高斯模糊 ----------
blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# ---------- Otsu 二值化 ----------
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("2. Otsu二值化结果 (铝制品应该是白色)", thresh)

# ---------- 形态学闭运算 ----------
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
cv2.imshow("3. 闭运算后 (白色应该是完整的圆)", closed)

# ---------- 找轮廓 ----------
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if not contours:
    print("❌ 没有找到任何轮廓！说明二值化完全失败了")
else:
    max_contour = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(max_contour)
    radius = int(radius * 0.95)
    center = (int(cx), int(cy))

    # 画出检测到的圆
    debug_img = img.copy()
    cv2.circle(debug_img, center, radius, (0, 255, 0), 3)
    cv2.circle(debug_img, center, 5, (0, 0, 255), -1)
    cv2.imshow("4. 检测到的圆 (绿圈应该贴合铝制品边缘)", debug_img)

    # 显示最终处理结果
    mask = np.zeros_like(gray)
    cv2.circle(mask, center, radius, 255, -1)
    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("5. 最终效果 (背景应该是纯黑)", result)

    print(f"✅ 检测到圆心: {center}, 半径: {radius}")
    print(f"   图片尺寸: {img.shape[1]} x {img.shape[0]}")
    print(f"   圆形占图片比例: {(radius*2/min(img.shape[:2])):.1%}")

print("\n按任意键关闭所有窗口...")
cv2.waitKey(0)
cv2.destroyAllWindows()
