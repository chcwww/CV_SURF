import cv2
import os
old_cwd = os.getcwd()
os.chdir(os.path.join(os.getcwd(), "CV_SURF"))
print(f"Working in '{os.getcwd()}'")
import surf_module

"""
Aim of this script is testing how SURF algorithm working
as a result wll be provided image with keypoints
"""

print(f"\n{'=' * 10} {'SURF START':^10} {'=' * 10}\n")

image_name = 'ntpu.jpg'
print(f"Image name : {image_name}")
image_path = os.path.join(os.getcwd(), 'images', 'ntpu', 'base.jpg')

img = cv2.imread(image_path)
print(f"Input image size : {img.shape}")

# Convert image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"Gray scale image size : {gray_image.shape}")

print(f"\n{'=' * 10} {'INIT':^10} {'=' * 10}\n")

surfObject = surf_module.SURF(threshold = 1000)
print(f"SURF detector create successfully...")
print(f"SURF detector initializing...")



# ================================
# DEBUGINGGGGGGGGGGGGGGGGGGGGGGGGG
# savep = os.path.join(os.getcwd(), 'output')
# testSURF = surf_module.SURF(threshold = 5000)
# test.border_size
# test.image = gray_image
# test.build_stretched_image()
# test.image_stretched.shape
# cv2.imwrite(savep + '\\stretched_3.png', test.image_stretched)

# test.build_integral_image()
# test.integral_image
# # cv2.imwrite(savep + '\\intergral_4.png', test.integral_image)

# test.compute_determinant_of_hessian()

# test.DetHes
# cv2.imwrite(savep + '\\dethes_4.png', test.DetHes[9])

# test.signLaplassian

# test.box_space_params # l0對到的其他參數

# testSURF.octaves_layers_L # 每層的l0長度







# DEBUGINGGGGGGGGGGGGGGGGGGGGGGGGG
# ================================



# Compute integral image and determinant of Hessian
surfObject.init(gray_image) 


print(f"\n{'=' * 10} {'KEYPOINT':^10} {'=' * 10}\n")

print("SURF detector detecting...")
# Find key point
keypoints, _ = surfObject.detectAndCompute()
print("Finish detecting...")
print(f'Keypoints len: {len(keypoints)}')

print(f"\n{'=' * 10} {'DRAWING':^10} {'=' * 10}\n")

print("Drawing keypoint plot...")
kp_image = cv2.drawKeypoints(img, keypoints, None, (0, 255, 0), 4)
#  flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
cv2.imwrite('output/naive_out.jpg', kp_image)

print(f"\n{'=' * 10} {'FINISH':^10} {'=' * 10}\n")