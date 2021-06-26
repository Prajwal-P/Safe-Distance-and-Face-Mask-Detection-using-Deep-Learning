# import the necessary packages
from .config import NMS_THRESHOLD, DETECTION_THRESHOLD, PEOPLE_COUNTER
import numpy as np
import cv2
import time
import os


def detect_people(frame, net, ln, personIdx=0):
    # grab the dimensions of the frame and  initialize the list of
    # results
    (H, W) = frame.shape[:2]
    results = []

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    end = time.time()
    print("Time taken to predict the image: {:.6f}seconds".format(end-start))
    layerOutputs = net.forward(ln)

    # initialize our lists of detected bounding boxes, centroids, and
    # confidences, respectively
    boxes = []
    centroids = []
    confidences = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter detections by (1) ensuring that the object
            # detected was a person and (2) that the minimum
            # confidence is met
            if classID == personIdx and confidence > DETECTION_THRESHOLD:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # centroids, and confidences
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(
        boxes, confidences, DETECTION_THRESHOLD, NMS_THRESHOLD)
    # print('Total people count:', len(idxs))
    # compute the total people counter
    if PEOPLE_COUNTER:
        human_count = "Human count: {}".format(len(idxs))
        cv2.putText(frame, human_count,
                    (frame.shape[0] - 170, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # update our results list to consist of the person
            # prediction probability, bounding box coordinates,
            # and the centroid
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)

    # return the list of results
    return results


if __name__ == "__main__":
    labelsPath = "./yolo-coco/coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")

    weightsPath = "./yolo-coco/yolov3.weights"
    configPath = "./yolo-coco/yolov3.cfg"
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # # sample image path
    # imgpath = './sample_dataset/1.jpg'
    for path in os.listdir('./sample_dataset'):  # loop to read one image at a time
        imgpath = os.path.join('./sample_dataset', path)
        image = cv2.imread(imgpath)
        image = cv2.resize(image, (720, 640))

        results = detect_people(
            image, net, ln, personIdx=LABELS.index("person"))

        # loop over the results
        for (i, (prob, bbox, centroid)) in enumerate(results):
            # extract the bounding box
            (startX, startY, endX, endY) = bbox

            # draw (1) a bounding box around the person and (2) the
            # centroid coordinates of the person,
            cv2.rectangle(image, (startX, startY),
                          (endX, endY), (0, 255, 255), 2)
            confidence = '{: .2f}%'.format(prob * 100)
            cv2.putText(image, confidence, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.circle(image, centroid, 1, (0, 0, 255), 2)
        cv2.imshow('image', image)

        # pauses for 3 seconds before fetching next image
        key = cv2.waitKey(2000)
        if key == 27:  # if ESC is pressed, exit loop
            cv2.destroyAllWindows()
            break
