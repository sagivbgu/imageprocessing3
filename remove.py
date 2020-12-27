import cv2

GREEN = (0, 255, 0)
RED = (0, 0, 255)

BACKGROUND_COLOR = 255


def remove_demarcation(img):
    contours, small_contours = get_contours_and_indexes_of_small_contours(img)

    erase_contours(img, small_contours, contours)

    # Now, we may still have some Kamatz-s left in the image
    # So, we want to get all the small components in the erased image and filter them out
    contours, small_contours = get_contours_and_indexes_of_small_contours(img)

    kamatzs = get_kamatzs(img, small_contours, contours)

    erase_contours(img, kamatzs, contours)

    return img


def get_contours_and_indexes_of_small_contours(img):
    binary_img = apply_otsu(img)

    contours = get_contours(binary_img)

    contours_by_height = sort_contours_by_height(contours)

    index_of_max_difference = get_index_of_max_difference(contours_by_height)

    # get "small" contours: all contours which are below the index of max difference
    small_contours = [index for index, size, cnt in contours_by_height[:index_of_max_difference]]

    # we filter-out all the contours that are much higher than their width
    small_contours = filter_small_contours_by_width(small_contours, contours)

    return contours, small_contours


def apply_otsu(img):
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img


def get_contours(img):
    # find only the extreme outer contours (connected components: letters and demarcation)
    # using chain approximation method
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def sort_contours_by_area_size(contours):
    # create a list of the indexes of the contours and their sizes
    contours_sizes = []
    for index, cnt in enumerate(contours):
        contours_sizes.append([index, cv2.contourArea(cnt), cnt])

    # sort them by the area size
    contours_sizes.sort(key=lambda x: x[1])

    return contours_sizes


def sort_contours_by_height(contours):
    contours_sizes = []
    for index, cnt in enumerate(contours):
        _, _, _, height = cv2.boundingRect(cnt)
        contours_sizes.append([index, height, cnt])

    # sort them by the height
    contours_sizes.sort(key=lambda x: x[1])

    return contours_sizes


def get_index_of_max_difference(contours_sizes):
    # In this method we iterate over the contours (sorted by area size)
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


def remove_small_contours(img, contours, contour_sizes, index_of_max_difference):
    for i in range(0, index_of_max_difference):
        erase_contour(img, contours, contour_sizes[i][0])


def erase_contours(img, indexes, contours):
    for index in indexes:
        erase_contour(img, contours, index)


def remove_patachs(img, indexes, contours):
    for index in indexes:
        if is_patach(contours[index]):
            print("Hi!")
            erase_contour(img, contours, index)
        else:
            pass

    return img


def erase_contour(img, contours, index):
    cv2.drawContours(img, contours, index, BACKGROUND_COLOR, -1)


def is_patach(contour):
    # assuming that there are no letters and other demarcations
    # which their width is twice as big as their height
    x, y, w, h = cv2.boundingRect(contour)
    if w >= 2 * h:
        return True
    else:
        return False


def get_kamatzs(img, indexes, contours):
    kamatzs = []
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for index in indexes:
        if is_kamatz(contours[index]):
            kamatzs.append(index)
            # cv2.drawContours(img, contours, index, GREEN, -1)
            # cv2.imshow("asd", img)
            # cv2.waitKey(0)
        else:
            pass
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return kamatzs


def is_kamatz(contour):
    # From all the small contours left, kamatz will have:
    #   1) a more complex polygon than other symbols (with at least 8-9 corners)
    #   2) the shape of the rectangle around it will be more square than others
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.005 * perimeter, True)
    _, _, w, h = cv2.boundingRect(contour)
    if 9 <= len(approx) and abs(w - h) <= 2:
        return True
    else:
        return False


def filter_small_contours_by_width(indexes, contours):
    new_indexes = []
    for index in indexes:
        _, _, w, h = cv2.boundingRect(contours[index])
        # if its much higher than its width, ignore it
        if h >= 1.8 * w:
            pass
        else:
            new_indexes.append(index)

    return new_indexes












# Extra Code
# loop through the contours, if the size of the contour is below a threshold,
    # draw a white shape over it in the input image
    # for cnt in contours:
    #     if cv2.contourArea(cnt) < 250:
    #         cv2.drawContours(img,[cnt],0,(120),-1)
    # display result

    # maxArea = 0
    # for cnt in contours:
    #     currArea = cv2.contourArea(cnt)
    #     if currArea > maxArea:
    #         maxArea = currArea
    #
    # print(maxArea)



    # sort the list based on the contour size.
    # this changes the order of the elements in the list

# img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

# matches = [cnt for cnt in contours if cv2.matchShapes(contours[0],cnt,1,0.0) < 0.1 and abs(cv2.contourArea(cnt) - cv2.contourArea(contours[0])) < 10]
# matches_res = [cv2.matchShapes(contours[0],cnt,1,0.0) for cnt in contours]

# for i in range(len(matches)):
#    # ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
#    cv2.drawContours(img, matches, i, GREEN, -1)

# cv2.drawContours(img, contours, 0, RED, -1)