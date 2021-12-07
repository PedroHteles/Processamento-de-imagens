# import the necessary packages
from filevideostream import FileVideoStream
from fps import FPS
import numpy as np
import argparse
import time
import cv2
from pyzbar.pyzbar import decode
from pyzbar.pyzbar import ZBarSymbol
from matplotlib import pyplot as plt


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to input video file")
args = vars(ap.parse_args())
# start the file video stream thread and allow the buffer to
# start to fill
print("[INFO] starting video file thread...")
fvs = FileVideoStream(args["video"]).start()
time.sleep(1.0)
# start the FPS timer
fps = FPS().start()

# loop over frames from the video file stream
while fvs.more():
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale (while still retaining 3
    # channels)
    frame = fvs.read()

    # frame = resize(frame, width=960)
    # frame = cv2.fastNlMeansDenoisingColored(frame,None,10,10,7,21)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(frame,(3,3),0)
    # laplacian = cv2.Laplacian(blur,cv2.CV_64F)
    # frame = laplacian/laplacian.max()
    # frame = np.dstack([frame, frame, frame])





    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    # ret, tess = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    frame = cv2.Laplacian(blur,cv2.CV_64F)
    
    # 

    # plt.subplot(2,2,4),plt.imshow(frame,cmap = 'gray')
    # plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

    # plt.show()


    cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)	

    barcodeList = decode(frame)
    for barcode in barcodeList:
        # (x, y, w, h) = barcode.rect
        (x, y, w, h) = barcode.rect
        barcodeData = barcode.data.decode("utf-8")
        print(barcodeData)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 15)            

    # show the frame and update the FPS counter
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
    fps.update()
    # stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
fvs.stop()