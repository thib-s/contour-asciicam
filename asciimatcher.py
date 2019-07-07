import argparse

import cv2
import numpy as np

CHAR_SHAPE = 22, 11  # 6, 11  # 12, 22


def generate_charmap(charset):
    charmap = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    linewidth = 1
    for c in charset:
        char_shape = cv2.getTextSize(c, font, scale, linewidth)[0]
        img = np.zeros((CHAR_SHAPE[0], CHAR_SHAPE[1]))
        charmap.append(
            {
                "template": cv2.blur(cv2.normalize(
                    cv2.putText(img, c, (0, CHAR_SHAPE[0] - abs(int((CHAR_SHAPE[0] - char_shape[1])*0.5))), font, scale, [255, 255, 255], linewidth, cv2.LINE_AA)
                , 0, 255, dtype=cv2.CV_8U),
                ksize=(3, 3)),
                "char": c
            }
        )
        # cv2.imshow(c, charmap[-1]["template"])
        # cv2.waitKey(100)
    return charmap


def image_to_string(img, charwitdth, charmap, alpha=0.3, display_canny=False):
    # print("charwidth:%s" % charwitdth)
    # print(img.shape)
    ratio = float(CHAR_SHAPE[1] * charwitdth) / img.shape[1]
    # print("ratio:%f" % ratio)
    rescaled = cv2.resize(img, (0, 0), fx=ratio, fy=ratio)
    # cv2.imshow("rescaled", rescaled)
    grayscale = cv2.cvtColor(rescaled, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", grayscale)
    # sobel = quick_sobel(*compute_gradients(grayscale))
    v = np.median(grayscale)
    lower = int(max(0, (1.0 - alpha) * v))
    upper = int(min(255, (1.0 + alpha) * v))
        # high_thresh, thresh_im = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sobel = cv2.Canny(grayscale, lower, upper)
    blurred = cv2.blur(sobel, (6, 6))
    if display_canny:
        cv2.imshow("canny", blurred)
        cv2.waitKey(10000)
    # matchings = {}
    n_col = int(sobel.shape[1] / CHAR_SHAPE[1])
    n_row = int(sobel.shape[0] / CHAR_SHAPE[0])
    # print("n_col:%s" % n_col)
    # print("n_row:%s" % n_row)
    assert n_col == charwitdth

    def compute_tile_dist(i, j, c):
        tile = blurred[i * CHAR_SHAPE[0]:(i + 1) * CHAR_SHAPE[0], j * CHAR_SHAPE[1]:(j + 1) * CHAR_SHAPE[1]]
        template = charmap[c]['template']
        return np.sum(np.square(tile - template))

    xx, yy, cc = np.meshgrid(range(n_row), range(n_col), range(len(charmap)))
    diffs = np.vectorize(lambda i, j, c: compute_tile_dist(i, j, c))(xx, yy, cc)
    matchings = np.argmin(diffs, axis=2)
    return matchings


def matchings_to_str(matchings, charmap, duplicate_spaces=1):
    chars = np.vectorize(lambda idx: charmap[idx]['char'])(matchings)
    lines = np.vectorize(lambda row: "".join(row), signature='(m)->()')(chars.transpose())
    text = "\n".join(lines)
    text = text.replace(" ", " "*duplicate_spaces)
    print(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='python asciimatcher.py path',
                                     description="convert image to ascii art by detecting it\'s contours")
    parser.add_argument(help='location of the image to convert',
                        dest="path")
    parser.add_argument("--outwidth",
                        help="width of the output image (number of chars)",
                        dest="outwidth",
                        default=80,
                        type=int)
    parser.add_argument("--charset",
                        help="list of allowed characters for the output string",
                        dest='charset',
                        default="abcdefghijklmnopqrstuvwxyz _-#()|0123456789*",
                        type=str,
                        )
    parser.add_argument("--canny_alpha",
                        dest="canny_alpha",
                        help="values between 0 and 1, the higher it is the more details there is",
                        type=float,
                        default=0.25)
    parser.add_argument("--duplicate_spaces",
                        help="repeat spaces as some devices shink those",
                        dest="duplicate_spaces",
                        default=1,
                        type=int)
    parser.add_argument("--display_canny",
                        help="if set, display the canny filter results for debugging",
                        action="store_true")
    args = parser.parse_args()
    img = cv2.imread(args.path)
    charset = args.charset
    charmap = generate_charmap(charset)
    matchings = image_to_string(img, args.outwidth, charmap, args.canny_alpha, args.display_canny)
    matchings_to_str(matchings, charmap, args.duplicate_spaces)
