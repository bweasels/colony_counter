import cv2
import os
import gc
import numpy
import math
import matplotlib
import sys
from optparse import OptionParser

matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt

# Defines how much darker a pixel needs to be to be considered significantly dark
# Smaller number = more selective (more false negatives)
# Larger number = more sensitive (more false positives)
SIG_THRESH_MULT = 0.4

# gradient resolution - multiplier to determine the gradient image size
# use a smaller number to avoid labeling colonies as background
# use a larger number to more accurately match a gradient
G_RES = 0.03

# X and Y dimensions of the shrunk and analyzed image
# Larger dimensions will make for a slower, more accurate analysis
# Smaller images may miss small colonies, but will be significantly faster
RESIZE_X = 1500
RESIZE_Y = 1000

# the smallest area (in um^2) that a colony has to be to be counted
COLONY_MIN_SIZE = 2000
# pixel to um scale - assuming a 4x image on the Nikon
# each pixel is 2.39um
# SCALE = 2.39

# pixel to um scale - assuming a 10x image on the Nikon
# each pixel is 3.38um
SCALE = 3.38

# Testing mode, stained variables, and magnification
TEST = False
STAINED = False
MAGNIFICATION = '4X'


def gradient_correction(img):
    # defining variables
    c_value = 128  # value to correct to to make a neutral image for QC
    c_map = []

    # Shrink and apply median blur to make a gradient image (g_img)
    g_img = cv2.resize(img, None, fx=G_RES, fy=G_RES, interpolation=cv2.INTER_AREA)
    g_img = cv2.medianBlur(g_img, 7)

    # only care about the first two values, height and width
    height, width = g_img.shape[:2]

    # make the correction map, a map of each pixel's deviation from the average
    for col in range(height):
        c_map.append([])
        for row in range(width):
            c_map[col].append(c_value - g_img[col][row])

    big_height, big_width = img.shape[:2]

    # apply the correction map to the corresponding pixel in the full sized image
    for col in range(big_height):
        for row in range(big_width):
            c_col = int(col * G_RES)
            c_row = int(row * G_RES)
            sig_thresh = g_img[c_col][c_row] * SIG_THRESH_MULT
            if img[col][row] > sig_thresh:  # reverse the < to undo comments
                img[col][row] = 0
            else:  # Set colonies to 0 for circle detect
                img[col][row] = 255

    del g_img
    del c_map

    return img


def contour_finding(thresh_img, img):
    shape_count = 0
    sum_area = 0

    # get the edges in the original image to help with inverted boundaries.
    # previously only looked for darker colonies - this looks for significantly lighter boundaries
    # BIG ISSUE## NEED TO TEST WITH ACTUAL BAD IMAGES
    edges = cv2.Canny(img, 50, 150)

    # add the canny detected edges to contours - edges that go nowhere will be noise
    thresh_img = cv2.add(thresh_img, edges)

    # produces the array of contours for the image
    img, contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # initialize an empty list to hold the colony sizes
    colonies = []
    selected = []

    # iterate through the contours and determine the area of each
    for c in range(len(contours)):
        area = cv2.contourArea(contours[c])
        # Rounding area for nicer output
        area = round(area, 2)
        # Area multiplied by scale^2 because units are pixels^2
        area = area * (SCALE ** 2)
        # most erroneous areas at this point smaller than 10 px
        if area > COLONY_MIN_SIZE:

            # fix any inward defects and get new area because colonies should be round
            smoothened = cv2.convexHull(contours[c])
            smoothArea = cv2.contourArea(smoothened)

            # add the shape to the counter and the lists
            shape_count = shape_count + 1
            sum_area = sum_area + smoothArea
            colonies.append(smoothArea)
            selected.append(smoothened)

    # draws the selected colonies
    if len(selected) != 0:
        cv2.drawContours(img, selected, -1, (255, 255, 0), 3)

    # draw the smallest size colony as a perfect circle
    minSizeRadius = int(math.sqrt(COLONY_MIN_SIZE / math.pi))
    cv2.circle(img, (minSizeRadius, minSizeRadius), minSizeRadius, (255, 255, 255), 3)

    # calculate avg_area
    avg_area = 0
    if shape_count > 0:
        avg_area = sum_area / shape_count

    # del thresh_img_2
    del thresh_img
    del contours

    return selected, shape_count, avg_area, colonies


def analyzeImgStained(img, root, shortName):
    # Shrink the image to a not-too-large, not-too-small size
    s_img = cv2.resize(img, (RESIZE_X, RESIZE_Y))
    print('Shrunk Image')

    # when image is passed as a parameter, its referenced, so it will get altered in
    # gradient_correction, so make a copy to retain the original shrunk image
    s_img_cp = s_img.copy()

    # Correct the background gradient to an even grey (127)
    gc_img = gradient_correction(s_img_cp)
    print('Corrected Gradient')

    # Use contours to select the largest colonies & draw the contours on the QC image (s_img)
    cont_img, count, avg_area, colonies = contour_finding(gc_img, s_img)
    print('Colonies counted')

    # producing the QC files filename with the proper path
    QCFolder = os.path.join(root, 'QC')
    QCFile = os.path.join(QCFolder, ('QC_' + shortName))

    # Save image to QC folder for QC
    cv2.imwrite(QCFile, cont_img)
    print('Saved as %s' % QCFile)
    print('~')

    # memory cleanup
    del img
    del gc_img
    del s_img
    del cont_img
    gc.collect()
    return count, avg_area, colonies


def analyzeImgStained(img, root, shortName):
    # Shrink the image to a not-too-large, not-too-small size
    s_img = cv2.resize(img, (RESIZE_X, RESIZE_Y))
    print('Shrunk Image')

    # when image is passed as a parameter, its referenced, so it will get altered in
    # gradient_correction, so make a copy to retain the original shrunk image
    s_img_cp = s_img.copy()

    # Correct the background gradient to an even grey (127)
    gc_img = gradient_correction(s_img_cp)
    print('Corrected Gradient')

    # Use contours to select the largest colonies & draw the contours on the QC image (s_img)
    cont_img, count, avg_area, colonies = contour_finding(gc_img, s_img)
    print('Colonies counted')

    # producing the QC files filename with the proper path
    QCFolder = os.path.join(root, 'QC')
    QCFile = os.path.join(QCFolder, ('QC_' + shortName))

    # Save image to QC folder for QC
    cv2.imwrite(QCFile, cont_img)
    print('Saved as %s' % QCFile)
    print('~')

    # memory cleanup
    del img
    del gc_img
    del s_img
    del cont_img
    gc.collect()
    return count, avg_area, colonies


def analyzeImg(img, root, shortName):
    # Shrink the image to a not-too-large, not-too-small size
    s_img = cv2.resize(img, (RESIZE_X, RESIZE_Y))
    print('Shrunk Image')

    # when image is passed as a parameter, its referenced, so it will get altered in
    # gradient_correction, so make a copy to retain the original shrunk image
    s_img_cp = s_img.copy()

    # Correct the background gradient to an even grey (127)
    gc_img = gradient_correction(s_img_cp)
    print('Corrected Gradient')

    if TEST:
        cv2.imshow('gradient corrected image', gc_img)

    # For unstained colonies################################
    # generates an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

    # First dialate, then erode (morph close), then dialate a whole bunch
    closed = cv2.morphologyEx(gc_img, cv2.MORPH_CLOSE, kernel)
    dialated = cv2.dilate(closed, kernel, iterations=4)
    ###########################################################
    if TEST:
        cv2.imshow('Denoised image', dialated)

    # Use contours to select the largest colonies & draw the contours on the QC image (s_img)
    selected, count, avg_area, colonies = contour_finding(dialated, s_img)
    QC_img = cv2.drawContours(cv2.resize(img, (RESIZE_X, RESIZE_Y)), selected, -1, (255, 255, 0), 3)
    print('Colonies counted')

    # producing the QC files filename with the proper path
    QCFolder = os.path.join(root, 'QC')
    QCFile = os.path.join(QCFolder, ('QC_' + shortName))

    # Save image to QC folder for QC
    if TEST:
        cv2.imshow('final QC Image', QC_img)
        cv2.waitKey()

    cv2.imwrite(QCFile, QC_img)
    print('Saved as %s' % QCFile)
    print('~')

    # memory cleanup
    del img
    del gc_img
    del s_img
    del QC_img
    gc.collect()
    return count, avg_area, colonies


def analyzeFolder(root, output):
    # make the directory for the QC files if it doesn't exist already
    QCPath = os.path.join(root, 'QC')
    if not os.path.exists(QCPath):
        os.makedirs(QCPath)
        print('Made QC folder.')

    folderCount = 0  # total number of colonies in this folder

    # iterate through the files in the folder and analyze
    fileList = os.listdir(root)
    for f in fileList:
        fullF = os.path.join(root, f)
        if f.endswith(".jpg"):
            # open the image
            img = cv2.imread(fullF, 0)
            print('Loaded %s' % f)

            # analyze image
            count, avg_area, colonies = analyzeImg(img, root, f)

            st_dev = numpy.std(colonies)
            # write colony counts to the output file
            output.write(f + ' colonies:,' + str(count) + '\n')
            # Write the average colony size and standard deviation to the output file
            output.write(
                f + ' average colony size and standard deviation (um^2):, ' + str('%.2f' % avg_area) + ',' + str(
                    '%.2f' % st_dev) + '\n')
            # Write the acutal list of colony sizes
            # ",".join(str etc... removes the brackets usually seen when printing out lists in python for easy graphing
            output.write(
                f + ' individual colony size (in no particular order):,' + ",".join(str(c) for c in colonies) + '\n')
            folderCount = folderCount + count  # Add each file's count to the folder overall count

    # write out the output
    output.write('Total folder count:' + str(folderCount) + '\n\n')

    return folderCount


def main():
    # use pareser to get commandline arguments
    parser = OptionParser()
    parser.add_option('-t', '--testing',
                      dest="TEST", action='store_true', default=False,
                      help='Testing mode will display the thresholded image, counted colonies, and final QC image for '
                           'each sample')
    parser.add_option('-m', '--magnification', dest='mag', default='4x',
                      help='Scope magnification - 4X or 10X. Default is 4X')
    parser.add_option('-s', '--stained', dest='stain', action='store_true', default=False,
                      help='Presence of Stain - staining yields more reliable numbers')
    parser.add_option('-i', '--input', dest='inputDir', default='.',
                      help='Directory to analyze images in. Defaults to current directory')

    # read out the options
    (options, args) = parser.parse_args()
    global TEST; TEST = options.TEST
    global STAINED; STAINED = options.stain
    global MAGNIFICATION; MAGNIFICATION = options.mag
    root = os.path.abspath(options.inputDir)

    containsJPG = False
    # open the output file start counter
    overallCount = 0  # total number of colonies
    outputFile = os.path.join(root, "output.csv")
    output = open(outputFile, "w+")  # create/open the output folder

    # The header displaying the parameters used in the analysis
    output.write(
        "The threshold for significance (percentage darker a colony has to be to be distinct from background) is: " +
        str(SIG_THRESH_MULT * 100) + "\n")
    output.write('The gradient map resolution multiplier is: ' + str(G_RES) + '\n')
    output.write(
        'This will produce a ' + str(RESIZE_X * G_RES) + ' by ' + str(RESIZE_Y * G_RES) + ' pixel gradient map.\n')
    output.write('Colonies smaller than ' + str(COLONY_MIN_SIZE) + ' um^2 are rejected as unviable or erroneous\n')
    output.write('------------------------------------------------------------------\n')

    # iterate through all subdirectories
    for dirName, subDirList, fileList in os.walk(root):

        for file in os.listdir(dirName):
            if file.endswith('.jpg'):
                containsJPG = True

        # exclude any existing quality control directories
        # these contain jpgs that we do not want to process
        if dirName.endswith('QC') is False and containsJPG is True:
            print('Analyzing Files in Directory: %s' % dirName)

            # if there are jpegs, begin analysis
            output.write(dirName + ': \n')
            folderCount = analyzeFolder(dirName, output)
            overallCount = overallCount + folderCount

            containsJPG = False  # reset for the next folder

    # write final lines and close to write out the buffer.
    output.write('\nTotal colony count: ' + str(overallCount))
    output.close()
    print("Images Analyzed Successfully!")
    return ()


if __name__ == "__main__":
    main()
