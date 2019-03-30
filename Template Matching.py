import cv2
import numpy as np

def DrawRectangles(img, minVals, scale, shapeTrgt ):
    for i in minVals:
        cv2.rectangle(img, ((int(i[1]/scale)) -1, int((i[0])/scale) -1), (int(i[1]/scale) + shapeTrgt[1], int(i[0]/scale) + shapeTrgt[0]), (255,255,0))


def TemplateMatchingImage(img_grey, trgt_grey, threshold):
    #Size img
    widthImg, heightImg = img_grey.shape
    widthTrgt, heightTrgt = trgt_grey.shape

    #Template matching image
    widthMatch = widthImg - widthTrgt + 1
    heightMatch = heightImg - heightTrgt + 1
    tempMatch = np.zeros((widthMatch,heightMatch), dtype=np.float32)

    isImgFound = False

    for i in range(0, widthMatch):
        for j in range(0, heightMatch):
            pixlSum = 0
            for x in range(0, widthTrgt):
                for y in range(0, widthTrgt):
                    matchPixl = (trgt_grey[x][y] - img_grey[i + x][j + y]) ** 2
                    pixlSum += matchPixl
            tempMatch[i][j] = pixlSum

    # Normaliize image
    tempMatch /= tempMatch.max()
    tempMatch *= 255

    minVals = []
    # Check if the image is found
    if tempMatch.min() / tempMatch.max() < threshold:
        isImgFound = True
        for i in range(0, widthMatch):
            for j in range(0, heightMatch):
                if tempMatch[i][j] == tempMatch.min():
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


def GetGreyScaled(img, target, scale):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    trgt_grey = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    if scale != 1.0:
        img_grey = cv2.resize(img_grey,(0,0),fx=scale,fy=scale);
        trgt_grey = cv2.resize(trgt_grey,(0,0),fx=scale,fy=scale);

    trgt_grey = trgt_grey.astype(np.float32)

    return img_grey, trgt_grey

def execute():
    # Load image and target image
    img = None
    target = None

    # User introduces image
    while img is None:
        img_dir = input('Set image : ')
        img = cv2.imread(img_dir)
        if img is None:
            print("Image not found, retry.")

    # User introduces target image
    while target is None:
        target_dir = input('Set target : ')
        target = cv2.imread(target_dir)
        if target is None:
            print("Target not found, retry.")

    # User introduces threshold
    threshold = float(input('Set threshold : (RECOMENDED: 0.01) '))

    scale = 0.0
    while scale == 0.0:
        scale = float(input(('Set scale : ')))
        if scale == 0.0:
            print("Scale can't be 0, retry.")

    img_grey, trgt_grey = GetGreyScaled(img, target, scale)

    normMatch, isSuccess, minVals = TemplateMatchingImage( img_grey, trgt_grey, threshold )

    outcmImg = DetermineOutcomeImage(isSuccess)

    if isSuccess:
        DrawRectangles(img, minVals, scale, target.shape)

    cv2.imshow("OUTCOME", outcmImg)

    # Show the image
    cv2.imshow("Target", target)
    cv2.imshow("Original", img)
    cv2.imshow("Mathing map", np.uint8(normMatch))
    cv2.waitKey(0)

execute()