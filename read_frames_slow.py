from fps import FPS
import numpy as np
import argparse
import imutils
import cv2


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


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to input video file")
args = vars(ap.parse_args())
stream = cv2.VideoCapture(args["video"])
fps = FPS().start()

# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video file stream
	(grabbed, frame) = stream.read()
	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break
	# resize the frame and convert it to grayscale (while still
	# retaining 3 channels)
	frame = resize(frame, width=450)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame = np.dstack([frame, frame, frame])
	# display a piece of text to the frame (so we can benchmark
	# fairly against the fast method)
	cv2.putText(frame, "Slow Method", (10, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)	
	# show the frame and update the FPS counter
	cv2.imshow("Frame", frame)
	cv2.waitKey(1)
	fps.update()