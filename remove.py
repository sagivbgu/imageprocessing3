import cv2

# This const is used to distinguish letters from demarcation - we assume letters are
# much higher than their width, i.e if h>=this_const * w => it's a letter
HEIGHT_TO_WIDTH_RATIO_THRESHOLD = 1.8

# This const is used to identify Kamatz.
# It is elaborated in the function: is_kamatz
KAMATZ_LEG_INDENT_RATIO = 0.25

# This const is used to identify Kamatz.
# Most Kamatzs have a shape of a square, this const sets the threshold of the difference
# between the height and the width of the expected kamatzs
KAMATZ_HEIGHT_TO_WIDTH_DIFFERENCE = 2

# This const is used to identify Kamatz.
# This const sets the expected number of corners a surrounding polygon of Kamatz
# will have. When using a polygon which its perimeter is close to the Kamatz's up to 0.5%
# of the Kamatz's perimeter, the typical number of corners is at least 8 (according to
# our observations)
KAMATZ_NUMBER_OF_POLYGON_CORNERS = 8

# Whether to blur the image before identifying demarcation. Use in cases where font is not "smooth" enough to prevent
# getting "holes" inside letters in the output.
# Default is False since in most cases it provides much better results.
BLUR_IMAGE = False

# Assuming background is white
BACKGROUND_COLOR = 255


def remove_demarcation(img):
    contours, small_contours = get_contours_and_indexes_of_small_contours(img)

    erase_contours(img, small_contours, contours)

    # Now, we may still have some Kamatz-s left in the image
    # So, we want to get all the small components in the erased image and filter them out
    contours, small_contours = get_contours_and_indexes_of_small_contours(img)

    kamatzs = get_kamatzs(small_contours, contours)

    erase_contours(img, kamatzs, contours)

    return img


def get_contours_and_indexes_of_small_contours(img):
    binary_img = apply_otsu(img)

    contours = get_contours(binary_img)

    contours_by_height = sort_contours_by_height(contours)

    index_of_max_difference = get_index_of_max_difference(contours_by_height)

    # get "small" contours: all contours which are below the index of max difference
    small_contours = [index for index, size in contours_by_height[:index_of_max_difference]]

    # we filter-out all the contours that are much higher than their width
    small_contours = filter_small_contours_by_width(small_contours, contours)

    return contours, small_contours


def apply_otsu(img):
    if BLUR_IMAGE:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img


def get_contours(img):
    # find both inner and outer contours (connected components: letters and demarcation)
    # using chain approximation method
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def sort_contours_by_height(contours):
    contours_sizes = []
    for index, cnt in enumerate(contours):
        _, _, _, height = cv2.boundingRect(cnt)
        contours_sizes.append([index, height])

    # sort them by the height
    contours_sizes.sort(key=lambda x: x[1])

    return contours_sizes


def get_index_of_max_difference(contours_sizes):
    # In this method we iterate over the contours (sorted by some size)
    # and for each two adjacent sizes in the list we calculate the difference.
    # Then, we determine the index in which the difference is the most significant.
    # Now we can assume: all the contours before this index are more likely to be the demarcation,
    # and all the contours after the index are more likely to be the letters
    index_of_max_difference = 0
    current_max_difference = 0
    for i in range(1, len(contours_sizes)):
        if contours_sizes[i - 1][1] != 0:
            size_difference = contours_sizes[i][1] / contours_sizes[i - 1][1]
            if size_difference > current_max_difference:
                current_max_difference = size_difference
                index_of_max_difference = i

    return index_of_max_difference


def filter_small_contours_by_width(indexes, contours):
    new_indexes = []
    for index in indexes:
        _, _, w, h = cv2.boundingRect(contours[index])
        # if its much higher than its width, ignore it
        if h >= HEIGHT_TO_WIDTH_RATIO_THRESHOLD * w:
            pass
        else:
            new_indexes.append(index)

    return new_indexes


def erase_contours(img, indexes, contours):
    for index in indexes:
        erase_contour(img, contours, index)


def erase_contour(img, contours, index):
    # erase the edges and a little bit outside the contour to avoid noise
    cv2.drawContours(img, contours, index, BACKGROUND_COLOR, 2)
    # erase the inner side of the contour
    cv2.drawContours(img, contours, index, BACKGROUND_COLOR, -2)


def get_kamatzs(indexes, contours):
    return [index for index in indexes if is_kamatz(contours[index])]


def is_kamatz(contour):
    # From all the small contours left, kamatz will have:
    #   1) a more complex polygon than other symbols (with at least 8 corners)
    #   2) the shape of the rectangle around it will be more square than others
    # To prevent misidentifying the letter Yod with Kamatz, we check if the most-right
    # point of the contour (A in example) is far enough from the most-right point in the bottom part
    # of the contour (B in example), i,e we check how much the leg of the symbol is "indented".
    # Example:
    #
    #   This is a typical shape of Yod:
    #
    #   * * * * A
    #   * * * * *
    #         * *
    #         * B
    #         * *
    #
    #   This is a typical shape of Kamatz:
    #
    #   * * * * * A
    #   * * * * * *
    #       * B
    #       * *
    #
    #   Usually, the x coordinate of A and B will be closer in Yod (the leg will be less indented)
    #   and further in Kamatz (more indented).
    #   The distance between them is relative to the width, and is set by the const KAMATZ_LEG_INDENT_RATIO
    #   which is set to 0.25 by default

    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.005 * perimeter, True)
    _, _, w, h = cv2.boundingRect(contour)

    most_right_x, _ = tuple(contour[contour[:, :, 0].argmax()][0])
    _, most_bottom_y = tuple(contour[contour[:, :, 1].argmax()][-1])
    # consider the bottom row a little bit higher than the actual bottom row
    # as we can see in the example - B is not in the most bottom row
    # this is used to generalize more fonts
    bottom_y = int(most_bottom_y - 0.25 * h)
    bottom_row_xs = [contour[i][0][0] for i in range(len(contour)) if
                     is_point_inside_contour((contour[i][0][0], bottom_y), contour)]
    most_right_x_in_bottom_row = max(bottom_row_xs)

    bottom_and_right_are_close = most_right_x - most_right_x_in_bottom_row < KAMATZ_LEG_INDENT_RATIO * w

    return len(approx) >= KAMATZ_NUMBER_OF_POLYGON_CORNERS \
           and abs(w - h) <= KAMATZ_HEIGHT_TO_WIDTH_DIFFERENCE \
           and not bottom_and_right_are_close


def is_point_inside_contour(point, contour):
    return cv2.pointPolygonTest(contour, point, False) >= 0
