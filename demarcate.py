from sys import argv
from remove import *


def load_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def save_image(img, path):
    cv2.imwrite(path, img)
    cv2.imshow("result", img)
    cv2.waitKey(0)


def main():
    if len(argv) != 3:
        print("Usage {0}: in_image out_image".format(argv[0]))
        exit()

    in_image = argv[1]
    out_image = argv[2]

    img = load_image(in_image)
    img = remove_demarcation(img)
    save_image(img, out_image)


if __name__ == "__main__":
    paths = ["article.jpg", "gil.jpg", "goodluck.png", "iloveimageprocessing.jpg",
             "large.jpg", "smaller.jpg", "tree.jpeg"]
    #paths = ["gil.jpg"]
    for path in paths:
        argv.append(path)
        argv.append("res_otsu_" + path)
        main()
        argv.pop()
        argv.pop()

