from inspect import unwrap
import cv2
import numpy as np
from asyncio.windows_events import NULL

from torch import cdist


def BLUR(frame, avr):
    kernel = np.ones((5, 5), np.float32)/avr
    dst = cv2.filter2D(frame, -1, kernel)
    return dst


def BLUR_NOISE(frame, kernel_size):
    dst = cv2.medianBlur(frame, kernel_size)
    return dst


def FIND_BIGGESTCONTOUR(contours, scale):
    maxArea = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        #print(str(area)+' type: '+str(type(area)))
        a = area-500*scale
        if a > 0:  # 20000:
            str1 = str(area)+' thres scale ='+str(area*scale)+'| '
            str2 = 'contours\'s len: '+str(len(contours))+'| '
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            points_number = len(approx)
            print(str1+str2 + str(points_number))
            # len(approx) == number of contour's points
            if(area > maxArea and len(approx) == 4):
                biggestContour = approx
            return biggestContour, cnt


def REORDER_POINTS(pts):
    pts = pts.reshape((4, 2))
    pts_new = np.zeros((4, 2), dtype=np.float32)

    add = pts.sum(axis=1)
    pts_new[0] = pts[np.argmin(add)]
    pts_new[2] = pts[np.argmax(add)]
    diff = np.diff(pts, axis=1)
    pts_new[1] = pts[np.argmin(diff)]
    pts_new[3] = pts[np.argmax(diff)]
    return pts_new


def CONTOUR_GET(scale, img, cap, bshowImg=True):
    scale *= 100
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # dilate and erode (connect gaps)
    dilate = cv2.dilate(img, np.ones((3, 3)), iterations=2)
    img = cv2.erode(dilate, np.ones((3, 3)), iterations=1)

    contour = img.copy()
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour = cv2.cvtColor(contour, cv2.COLOR_GRAY2BGR)
    biggest, biggest_cnt = FIND_BIGGESTCONTOUR(contours, scale)

    if biggest.size != 0:
        if bshowImg:
            contour = cap.copy()
        cv2.drawContours(contour, biggest_cnt, -1, (0, 255, 255), 5)
        x, y, w, h = cv2.boundingRect(biggest)
        cv2.rectangle(contour, (x, y), (x+w, y+h), (255, 255, 165), 3)
        cv2.putText(contour, 'Area: '+str(cv2.contourArea(biggest)), (x+0, y+5),
                    cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(contour, ' points: '+str(len(biggest)), (x+0, y+60),
                    cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 255, 0), 2)
        print('pre-reorder: \n'+str(biggest))
        biggest = REORDER_POINTS(biggest)
        print('after-reorder: \n'+str(biggest))
    return contour, biggest


def UNWARP(cap, biggest):
    img = cap.copy()
    rect = biggest
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, matrix, (maxWidth, maxHeight))
    print('dst: \n'+str(dst)+'\nwarped: \n'+str(matrix))
    cv2.imshow('warped', warped)
    return warped


def PAPER_EFFECT(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_binImg = cv2.adaptiveThreshold(frame, 255, 1, 1, 7, 2)
    # make all 1 to 0, make all 0 to 1
    frame_binImg = cv2.bitwise_not(frame_binImg)
    frame_binImg = cv2.medianBlur(frame_binImg, 3)
    cv2.imshow('effect', frame_binImg)
    return frame_binImg


def SHARP(frame):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    dst = cv2.filter2D(frame, -1, kernel)
    return dst


def EDGE(frame, blurG=NULL, kernelGauss=5, threshold=(50, 150)):
    # get grayScale -> GaussBlur -> edge use Canny:
    gs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernelGauss = 5
    kernel = (kernelGauss, kernelGauss)
    if blurG.all() == NULL:
        blurG = cv2.GaussianBlur(gs, kernel, 5)
    low_threshold = threshold[0]
    high_threshold = threshold[1]
    edge = cv2.Canny(blurG, low_threshold, high_threshold)
    return edge


def HOUGHLINE(frame, edge):
    edgeline = np.copy(frame)*0
    edge = cv2.cvtColor(edge, cv2.COLOR_BGR2GRAY)
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    # minimum number of votes (intersections in Hough grid cell)
    threshold = 15
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    cdstP = edge.copy()
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edge, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    cdstP = cv2.cvtColor(cdstP, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]),
                     (255, 255, 0), 3, cv2.LINE_AA)
    # Draw the lines on the  image
    lines_edges = cv2.addWeighted(frame, 0.8, edgeline, 1, 0)
    #cv2.imshow('HoughLine', lines_edges)
    return cdstP
    # return lines_edges


def MULTI_DISPLAY(scale, original, grey_3_channel, blur=1, threshold=1, contour=1, warp_perspective=1, blank=NULL):
    frameArr = [original, grey_3_channel, blur,
                threshold, contour, warp_perspective]
    idx = 0
    for x in frameArr:
        if type(x) is type(original):
            frameArr[idx] = cv2.resize(x, (0, 0), None, scale, scale)
            if idx == 1:
                w = int(frameArr[0].shape[1])
                h = int(frameArr[0].shape[0])
                blank = cv2.resize(np.zeros((w, h, 3), np.uint8), (w, h))
        if type(x) is int:
            frameArr[idx] = np.array(blank)
        # print('frame['+str(idx)+'] stat: '+str(type(frameArr[idx]))
        #       + 'Dimension: '+str(frameArr[idx].shape[0])+'-'
        #       + str(frameArr[idx].shape[1]))
        idx += 1
    print('------')
    matrixH1 = np.hstack((frameArr[0], frameArr[1], frameArr[2]))
    matrixH2 = np.hstack((frameArr[3], frameArr[4], frameArr[5]))
    matrix_Total = np.vstack((matrixH1, matrixH2))
    return matrix_Total


def IMG(scale, imgPath, sigmaG=5, canny_thres=(50, 165), bGetContour=False):
    cap = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    cap = cv2.resize(cap, (0, 0), None, scale, scale)
    gs = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
    blur = cv2.cvtColor(cv2.GaussianBlur(
        gs, (9, 9), sigmaG), cv2.COLOR_GRAY2BGR)
    edge = cv2.cvtColor(
        EDGE(cap, blur, sigmaG, canny_thres), cv2.COLOR_GRAY2BGR)
    line = HOUGHLINE(cap, edge)
    if bGetContour:
        contour, biggestContour = CONTOUR_GET(scale, line, cap)
        unwrap = UNWARP(cap, biggestContour)
        contour2, biggestContour = CONTOUR_GET(scale, line, cap, False)
        return MULTI_DISPLAY(scale, cap, blur, edge, line, contour2, contour), unwrap
    return MULTI_DISPLAY(scale, cap, edge, line), cap


def VID_1(cap, scale=.5, kernel_Gauss=(3, 3), kernel_Noise=29, canny_thres=(160, 160)):
    width = int(cap.get(3))
    height = int(cap.get(4))
    print('2nd w+h: '+str(width) + ' '+str(height))
    # define the codec of the output vid( if any)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('grayOut.mp4', fourcc, 10, (width, height))
    if not(cap.isOpened()):
        print('Error: cant load vid')
    elif(cap.isOpened()):
        ret, frame = cap.read()
        keypress = cv2.waitKey(5) & 0xFF
        gs_3c = cv2.cvtColor(cv2.cvtColor(
            frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        return MULTI_DISPLAY(scale, frame, gs_3c)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
