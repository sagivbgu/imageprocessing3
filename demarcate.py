from sys import argv
import cv2

from remove import remove_demarcation


def load_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def save_image(img, path):
    cv2.imwrite(path, img)


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
    main()
