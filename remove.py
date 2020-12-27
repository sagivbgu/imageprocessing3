import cv2

GREEN = (0, 255, 0)
RED = (0, 0, 255)


def remove_demarcation(img):
    thres = apply_otsu(img)

    # find contours in the thresholded image (this gives all symbols)
    contours, hierarchy = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # loop through the contours, if the size of the contour is below a threshold,
    # draw a white shape over it in the input image
    # for cnt in contours:
    #     if cv2.contourArea(cnt) < 250:
    #         cv2.drawContours(img,[cnt],0,(120),-1)
    # display result

    maxArea = 0
    for cnt in contours:
        currArea = cv2.contourArea(cnt)
        if currArea > maxArea:
            maxArea = currArea

    print(maxArea)

    # create a list of the indexes of the contours and their sizes
    contour_sizes = []
    for index, cnt in enumerate(contours):
        contour_sizes.append([index, cv2.contourArea(cnt)])

    # sort the list based on the contour size.
    # this changes the order of the elements in the list
    contour_sizes.sort(key=lambda x: x[1])

    # loop through the list and determine the largest relative distance
    indexOfMaxDifference = 0
    currentMaxDifference = 0
    for i in range(1, len(contour_sizes)):
        if contour_sizes[i - 1][1] != 0:
            sizeDifference = contour_sizes[i][1] / contour_sizes[i - 1][1]
            if sizeDifference > currentMaxDifference:
                currentMaxDifference = sizeDifference
                indexOfMaxDifference = i

    background_color = 255

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    matches = [cnt for cnt in contours if cv2.matchShapes(contours[0],cnt,1,0.0) < 0.1 and abs(cv2.contourArea(cnt) - cv2.contourArea(contours[0])) < 10]
    matches_res = [cv2.matchShapes(contours[0],cnt,1,0.0) for cnt in contours]

    for i in range(len(matches)):
        # ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
        cv2.drawContours(img, matches, i, GREEN, -1)

    cv2.drawContours(img, contours, 0, RED, -1)

    # loop through the list again, ending (or starting) at the indexOfMaxDifference, to draw the contour
    for i in range(0, indexOfMaxDifference // 2):
        pass #cv2.drawContours(img, contours, contour_sizes[i][0], background_color, -1)

    return img


def apply_otsu(img):
    # THIS IS NOT OTSU
    # convert to binary. Inverted, so you get white symbols on black background
    #_, thres = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    # Otsu's thresholding
    _, thres = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thres
