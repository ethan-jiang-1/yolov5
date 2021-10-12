from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

import os
import cv2
import time
from collections import namedtuple
from collections import deque

ObjTrack = namedtuple("ObjTrack", ["centroid", "obj_klx", "conf", "rect", "area"])


TRACK_AVE_WEIGHT_DECAY = 10  # the higher the smother


class CentroidTracker():
    def __init__(self, maxDisappeared=50):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.objklxs = OrderedDict()
        self.disappeared = OrderedDict()

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

    def register(self, centroid, klx):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.objklxs[self.nextObjectID] = klx
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rect_klxs):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rect_klxs) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # return early as there are no centroids or tracking info
            # to update
            return self.objects, self.objklxs

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rect_klxs), 2), dtype="int")
        inputCentrklxs = np.zeros((len(rect_klxs)), dtype="int")

        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY, objKlx)) in enumerate(rect_klxs):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
            inputCentrklxs[i] = objKlx

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], inputCentrklxs[i])

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], inputCentrklxs[col])

        # return the set of trackable objects
        return self.objects, self.objklxs


class DetectionTracker(object):
    def __init__(self, ch, cw, maxDisappeared=100, maxQueueLen=5):
        self.maxDisappeared = maxDisappeared
        self.maxQueueLen = maxQueueLen
        self.ch = ch
        self.cw = cw
        
        self.cur_objs = {}
        self.cur_tracks = []

        self.tracked_objs_history = deque(maxlen=maxQueueLen)
        self.tracked_tracks_history = deque(maxlen=maxQueueLen)
        self.cur_frame_no = -1

        self.traker = CentroidTracker(maxDisappeared=maxDisappeared)

    def update(self, rect_klxs, tracks):
        self.cur_frame_no += 1
        objects, objklxs = self.traker.update(rect_klxs)
        if len(self.cur_objs) != 0:
            self.tracked_objs_history.append(self.cur_objs)
            self.tracked_tracks_history.append(self.cur_tracks)
        self.cur_tracks = tracks

        self.cur_objs = {}
        for (objectID, centroid) in objects.items():
            klx = objklxs[objectID]

            track_found = self._find_track(centroid, tracks=tracks)

            if track_found is not None:
                self.cur_objs[objectID] = (centroid, track_found, klx)
            else:
                self.cur_objs[objectID] = (centroid, None, klx)
        return self.cur_objs

    def _find_track(self, centroid, tracks=None):
        if tracks is None:
            tracks = self.cur_tracks
        for track in tracks:
            if track.centroid[0] == centroid[0] and track.centroid[1] == centroid[1]:
                return track
        return None

    def _convert_xywh2rectca(self, x, y, w, h, klx):
        cw = self.cw 
        ch = self.ch
        x0 = int((x - 0.5 * w) * cw + 0.5)
        x1 = int((x + 0.5 * w) * cw + 0.5)
        y0 = int((y - 0.5 * h) * ch + 0.5)
        y1 = int((y + 0.5 * h) * ch + 0.5)

        cX = int((x0 + x1) / 2.0)
        cY = int((y0 + y1) / 2.0)
        return [x0, y0, x1, y1, klx], (cX, cY), (x1 - x0) * (y1 - y0)

    def feed_detects(self, detections):
        rect_klxs = []
        tracks = []
        for detection in detections:
            obj_klx, x, y, w, h, conf = detection

            rect_klx, centroid, area = self._convert_xywh2rectca(x, y, w, h, obj_klx)
            rect_klxs.append(rect_klx)
            tracks.append(ObjTrack(centroid=centroid, obj_klx=obj_klx, conf=conf, rect=rect_klx[0:4], area=area))

        return rect_klxs, tracks

    def _get_tracked_xywh(self, objectID, tracks=None, objs=None):
        if tracks is None:
            tracks = self.cur_tracks
        if objs is None:
            objs = self.cur_objs

        if objectID not in objs:
            return None

        _, track_found, _ = objs[objectID] 
        if track_found is None:
            return None

        rect = track_found.rect
        xywh = track_found.centroid[0], track_found.centroid[1], rect[2] - rect[0], rect[3] - rect[1]
        return xywh 

    def get_tracked_txywh(self, objectID):
        xywhs = []
        xywh_now = self._get_tracked_xywh(objectID, tracks=None, objs=None)
        if xywh_now is not None:
            xywhs.append(xywh_now)

            for i in range(len(self.tracked_tracks_history)):
                tracks = self.tracked_tracks_history[i]
                objs = self.tracked_objs_history[i]
                xywh_prev = self._get_tracked_xywh(objectID, tracks=tracks, objs=objs)
                if xywh_prev is not None:
                    xywhs.append(xywh_prev)
        
        if len(xywhs) == 0:
            return None
        elif len(xywhs) == 1:
            return xywhs[0]

        total_weight = 0
        tx, ty, tw, th = 0, 0, 0, 0
        for i in range(len(xywhs)):
            x, y, w, h = xywhs[i]

            weight = (1 - (1 / self.maxQueueLen * i) / TRACK_AVE_WEIGHT_DECAY)
            total_weight += weight
            tx += x * weight
            ty += y * weight
            tw += w * weight 
            th += h * weight 

        txywh = (int(tx/total_weight), int(ty/total_weight), int(tw/total_weight), int(th/total_weight))
        return txywh

    def draw_traced_detections(self, detections, fun_draw_tracked_fun=None, img=None, extra=None):
        # detections array of [obj_klx, x, y, w, h, conf]
        rect_klxs, tracks = self.feed_detects(detections)

        tracked_objs = self.update(rect_klxs, tracks)

        for objectID, val in tracked_objs.items():
            centroid, track_found, klx = val
            avg_xywh = self.get_tracked_txywh(objectID)

            if fun_draw_tracked_fun is not None:
                fun_draw_tracked_fun(objectID, track_found, centroid, klx, avg_xywh, img=img, extra=extra)


def _get_txt_img_names(tracking_txt_dir, tracking_img_dir):
    tracking_txt_dir = os.path.abspath(tracking_txt_dir)
    names = os.listdir(tracking_txt_dir)
    names = sorted(names)
    fnames_txt = []
    for name in names:
        if name.endswith(".txt"):
            fnames_txt.append("{}/{}".format(tracking_txt_dir, name))

    tracking_img_dir = os.path.abspath(tracking_img_dir)
    names = os.listdir(tracking_img_dir)
    names = sorted(names)
    fnames_img = []
    for name in names:
        if name.endswith(".jpg"):
            fnames_img.append("{}/{}".format(tracking_img_dir, name))
    return fnames_txt, fnames_img

def _read_dections(fntxt):
    detections = []
    with open(fntxt, "rt") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            params = line.split(" ")
            obj_klx, x, y, w, h = int(params[0]), float(params[1]), float(params[2]), float(params[3]), float(params[4])
            conf = 1
            if len(params) >= 6:
                conf = float(params[5])
            detections.append((obj_klx, x, y, w, h, conf))
    return detections

def _fun_draw_tracked_fun(objectID, track_found, centroid, klx, avg_xywh, img=None, extra=None):
    frame = img
    if frame is None:
        return 

    if track_found is not None:
        color = (0, 0, 255)
        text = "Id{}/{} {:.2f}".format(objectID, klx, track_found.conf)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1)
        cv2.circle(frame, (centroid[0], centroid[1]), 2, color, -1)

        if avg_xywh is not None:
            ax, ay, aw, ah = avg_xywh
            pt1 = (int(ax - 0.5 * aw), int(ay - 0.5 * ah))
            pt2 = (int(ax + 0.5 * aw), int(ay + 0.5 * ah))
            color_avg = (255, 255, 255)
            cv2.rectangle(frame, pt1, pt2, color_avg, thickness=2)
            cv2.circle(frame, (ax, ay), 4, color_avg, -1)
    else:
        color = (196, 196, 196)
        text = "id{}/{}".format(objectID, klx)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, color, -1)


def _test_draw_detections_from_files(tracking_txt_dir, tracking_img_dir):
    fnames_txt, fnames_img = _get_txt_img_names(tracking_txt_dir, tracking_img_dir)

    img = cv2.imread(fnames_img[0], cv2.IMREAD_COLOR)
    ch, cw = img.shape[0], img.shape[1]
    dtt = DetectionTracker(ch, cw)

    for fntxt, fnimg in zip(fnames_txt, fnames_img):
        img = cv2.imread(fnimg, cv2.IMREAD_COLOR)

        detections = _read_dections(fntxt)

        dtt.draw_traced_detections(detections, fun_draw_tracked_fun=_fun_draw_tracked_fun, img=img, extra=None)

        cv2.imshow("Frame", img)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        time.sleep(1/30)
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    data_root = os.path.dirname(__file__)

    tracking_txt_dir = "{}/../../dx_mp4_tracking/t0_labels".format(data_root)
    tracking_img_dir = "{}/../../dx_mp4_tracking/t0_images".format(data_root)
    _test_draw_detections_from_files(tracking_txt_dir, tracking_img_dir)
