import cv2
import numpy as np

def DrawRectangles(img, tmpl, TempMatch, minVals):
    # Size img
    rowsTmpl, colsTmpl, chan = tmpl.shape
    rowsMatch, colsMatch = TempMatch.shape

    rowHalf = np.int8(rowsTmpl/2)
    colHalf = np.int8(colsTmpl/2)

    for i in minVals:
        cv2.rectangle(TempMatch, (i[0] - rowHalf, i[1] - colHalf), (i[0] + rowHalf, i[1] + colHalf), (255,255,0))


def TemplateMatchingImage(imgStr, tmplStr):
    img = cv2.imread(imgStr, cv2.IMREAD_GRAYSCALE)
    tmpl = cv2.imread(tmplStr, cv2.IMREAD_GRAYSCALE)
    img = np.float64(img)
    tmpl = np.float64(tmpl)

    #Size img
    rowsImg, colsImg = img.shape
    rowsTmpl, colsTmpl = tmpl.shape

    #Template matching image
    tempMatch = np.zeros((rowsImg - rowsTmpl + 1, colsImg - colsTmpl + 1))
    rowsMatch, colsMatch = tempMatch.shape

    isImgFound = False

    for i in range(0, rowsMatch):
        for j in range(0, colsMatch):
            pixlSum = 0
            for x in range(0, rowsTmpl):
                for y in range(0, colsTmpl):
                    matchPixl = (tmpl[x][y] - img[i + x][j + y]) ** 2
                    pixlSum += matchPixl
            tempMatch[i][j] = pixlSum

    # Check if the image is found
    if tempMatch.min() / tempMatch.max() < 0.1:
        isImgFound = True

    # Normaliize image
    tempMatch /= tempMatch.max()
    tempMatch *= 255

    minVals = []
    blackest = 100
    for i in range(0, rowsMatch):
        for j in range(0, colsMatch):
            if tempMatch[i][j] <= tempMatch.min():
                minVals.append([i, j])

    return tempMatch, isImgFound, minVals


def DetermineOutcomeImage(isSuccess):
    if isSuccess:
        return MakeLabel((40, 245, 3), "TARGET FOUND", (0, 255, 0))
    else:
        return MakeLabel((40, 320, 3), "TARGET NOT FOUND", (0, 255, 255))

def MakeLabel(size, text, colour):
    #Create a black image
    lblImg = np.zeros(size, np.uint8)

    #Write the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(lblImg, text, (5,30), font, 1, colour, 2)

    return lblImg


def execute():
    # Load image and
    img = cv2.imread("img2.png")
    tmpl = cv2.imread("t1-img2.png")

    TempMatch, isSuccess, minVals = TemplateMatchingImage("img2.png", "t1-img2.png")

    outcmImg = DetermineOutcomeImage(isSuccess)

    if isSuccess:
        DrawRectangles(img, tmpl, TempMatch, minVals)

    cv2.imshow("OUTCOME", outcmImg)

    # Show the image
    cv2.imshow("A", np.uint8(TempMatch))
    cv2.imshow("Original", img)
    cv2.imshow("Filtered", tmpl)
    cv2.waitKey(0)

execute()