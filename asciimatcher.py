import cv2
import numpy as np

COL = 16
ROW = 5
CHAR_SHAPE = 12, 22#6, 11  # 12, 22
# IMG = cv2.imread("ASCII_full_adjusted_small.png")[:, :, 0]
IMG = cv2.imread("ASCII_full_adjusted.png")[:, :, 0]
IMG_BLURRED = cv2.blur(IMG, (7, 7))
# COL = 37
# ROW = 1
# CHAR_SHAPE = 8, 10
# IMG = cv2.imread("ASCII.jpg")[:, :, 0]
IMG_MATRIX = [IMG_BLURRED[CHAR_SHAPE[1] * i:CHAR_SHAPE[1] * (i + 1), CHAR_SHAPE[0] * j:CHAR_SHAPE[0] * (j + 1)] for j in
              range(COL) for i in range(ROW)]
IMG_MATRIX_UNBLURRED = [IMG[CHAR_SHAPE[1] * i:CHAR_SHAPE[1] * (i + 1), CHAR_SHAPE[0] * j:CHAR_SHAPE[0] * (j + 1)] for j
                        in
                        range(COL) for i in range(ROW)]


# for i in range(COL*ROW):
#     cv2.imshow("char0", IMG_MATRIX[i])
#     if cv2.waitKey(200) & 0xFF == ord('q'):
#         break

def find_matching_char(patch):
    if patch.shape != (CHAR_SHAPE[1], CHAR_SHAPE[0]):
        return np.zeros(patch.shape)
    # patch = cv2.resize(patch, (0, 0), None, 8, 10, cv2.INTER_LINEAR)
    best_score = np.inf
    # cv2.imshow("test_patch", patch)
    # cv2.waitKey(1000)
    for i in range(ROW * COL):
        score = np.sum(np.square(patch - IMG_MATRIX[i]))
        if score < best_score:
            best = np.copy(IMG_MATRIX_UNBLURRED[i])
            best_score = score
    return best


def transform_image(img):
    img = img[:, :, 0].copy()
    # img = cv2.Canny(img, 100, 120)
    img = cv2.blur(cv2.Canny(img, 100, 120), (12, 12))
    cv2.imshow("canny", img)
    for i in range(0, img.shape[0], CHAR_SHAPE[1]):
        for j in range(0, img.shape[1], CHAR_SHAPE[0]):
            img[i:i + CHAR_SHAPE[1], j:j + CHAR_SHAPE[0]] = find_matching_char(
                img[i:i + CHAR_SHAPE[1], j:j + CHAR_SHAPE[0]])
    return img

if __name__ == '__main__':
    img = cv2.imread("quadripod.png")
    res = transform_image(cv2.resize(img, (0, 0), fx=1, fy=1))
    cv2.imshow("transformed", res)
    cv2.waitKey(10000)
    cv2.imwrite("ascii_quadripod.png", res)