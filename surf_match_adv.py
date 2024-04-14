import cv2
import os
old_cwd = os.getcwd()
os.chdir(os.path.join(os.getcwd(), "CV_SURF"))
print(f"Working in '{os.getcwd()}'")
import surf_module

"""
Input : base_image and other test image
Output : image with keypoint for each file and Comparation of base image and other
transformed images (testing Scale Invariant and Rotation Invariant)
"""


image_set = 'ntpu'

input_path = 'images/' + image_set + '/'
output_path = 'ntpu_result/'

file_list = os.listdir(input_path)
print(f'Test files: {file_list}')

# =======Preprocessing=======
pre = 0
if pre :
    file_name = 'base.jpg' # base.jpg
    img_base = cv2.imread(input_path + file_name)
    img_size = img_base.shape
    img_reduce = cv2.resize(img_base, (img_size[1] // 3, img_size[0] // 3))
    cv2.imwrite(input_path + file_name, img_reduce)

    # rotate
    img_base = cv2.imread(input_path + file_name)
    (h, w) = img_base.shape[:2]
    center = (w  / 2, h / 2)

    for angle in [-90, -30, 30, 90, 180] :
        scale = 1 if abs(angle) == 30 else 1
        M = cv2.getRotationMatrix2D(center, angle, scale)
        img_rot = cv2.warpAffine(img_base, M, (w, h), borderMode = cv2.BORDER_CONSTANT)
        # cv2.BORDER_TRANSPARENT cv2.BORDER_REPLICATE
        cv2.imwrite(input_path + f"rot_{angle}.jpg", img_rot)

#============================


matcher = surf_module.SURF_Matcher()

print(f'Processing original image...')
file_name = 'base.jpg' # base.jpg
img_base = cv2.imread(input_path + file_name)

gray_image_base = cv2.cvtColor(img_base, cv2.COLOR_BGR2GRAY)

surfObject = surf_module.SURF(threshold=1000)
surfObject.init(gray_image_base)

keypoints_base, descriptors_base = surfObject.detectAndCompute()

# Print number of keypoints and descriptors
print(f'Original keypoints len: {len(keypoints_base)}')
print(f'Original descriptors len: {len(descriptors_base)}')

# surf_module.save_keypoints_to_file(keypoints_base, output_path + 'image_base_kp.txt')
# surf_module.save_descriptors_to_file(descriptors_base, output_path + 'image_base_des.txt')

kp_image = cv2.drawKeypoints(img_base, keypoints_base, None, color=(
    0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite(output_path + 'image_base_kp.jpg', kp_image)

print(f'Original image finish detecting\n')

for file in file_list:
    if file == 'base.jpg':
        continue

    print(f'Processing {file}')

    filename = os.path.splitext(file)[0]

    img = cv2.imread(input_path + file)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    surfObject_file = surf_module.SURF(threshold=1000)
    surfObject_file.init(gray_image)

    keypoints, descriptors = surfObject_file.detectAndCompute()

    # Print number of keypoints and descriptors
    print(f'Test keypoints len {filename}: {len(keypoints)}')
    print(f'Test keypoints len {filename}: {len(descriptors)}')

    # surf_module.save_keypoints_to_file(keypoints_base, output_path + 'image_' + filename + '_kp.txt')
    # surf_module.save_descriptors_to_file(descriptors_base, output_path + 'image_' + filename + '_des.txt')

    kp_image = cv2.drawKeypoints(img, keypoints, None, color=(
        0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(output_path + 'image_' + filename + '_kp.jpg', kp_image)

    print(f'Matching...')

    matches = matcher.match(descriptors_base, descriptors)

    matches = sorted(matches, key=lambda x: x.distance)
    # Draw first 20 matches. (距離最小那些)
    img_match = cv2.drawMatches(img_base, keypoints_base, img, keypoints, matches[:50], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imwrite(output_path + 'image_' + filename + '_match.jpg', img_match)

    print(f'Match of {file} finish\n')

print('Finish...')
