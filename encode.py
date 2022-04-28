import torch
import cv2
import pandas as pd
import numpy as np

from autoencoder.autoencoder import Autoencoder
from utils.model_utils import Transformer
from utils.utils import nms_counts
from utils.encode_utils import *


class ReverseEncoder():
    """Gets worms of interest from yolo detections at the frame specified by sort.
    Also gets the outputs from the classification. For any missed calls it'll run
    back with the autoencoder to updated the list of worm calls.
    """
    weights_path = "autoencoder/weights/autoencoder_4.pt"
    img_size = 28
    transformer = Transformer(thresh_use=True, square_use=True, img_size=img_size)

    def __init__(self, sort_output, yolo_csv: str, video_path: str, end=False, device="cpu"):
        """
        Args:
            sort_output (pd DF | str): Analyze sort output.
            yolo_csv (str): Path to yolo detections.
            video_path (str): Path to video avi.
            device (str, optional): Torch device. Defaults to "cpu".
        """
        self.sort_output = sort_output
        self.df = pd.read_csv(yolo_csv,
                              usecols=[0, 1, 2, 3, 4, 5],
                              names=["frame", "x", "y", "w", "h", "class"])
        self.video_path = video_path
        self.device = device

        self.encoder = Autoencoder().to(device)
        self.encoder.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        print(f"Encoder -> Loaded weights from {self.weights_path} \nTo device: {self.device}")

        # Declare video and organize sort output.
        self.video = cv2.VideoCapture(self.video_path)
        self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

        if end:
            self.end = end
            if self.end >= self.frame_count:
                self.end = self.frame_count - 1
        else:
            self.end = self.frame_count

        if type(self.sort_output) == str:
            self.sort_output = pd.read_csv(self.sort_output)

        # Worms of interest that are not picked up by sort.
        self.woi = self.declare_woi(window=10, min_count=2, nms=0.7)

    def get_worms_from_frame(self, frame_id):
        """ Gets the frame image from a frame id,
        and then the bounding boxes associated with that image"""
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
        ret, frame = self.video.read()  # Read frame
        bbs = self.df[self.df["frame"] == frame_id]
        bbs = bbs.to_numpy()
        bbs = bbs[:, 1:5]
        return frame, bbs

    def declare_woi(self, window=10, min_count=2, nms=0.7):
        """Declares worms of interest at the point where sort makes the last
        time of death call. Compares this to already called points by sort
        and determines which worms need to be tracked by encoder.
        """
        min_distance = 40  # Min distance to match sort to new woi.
        first = self.end - window
        last = self.end

        all_bbs = np.empty((0, 4), int)
        for i in range(first, last):
            _, bbs = self.get_worms_from_frame(i)
            all_bbs = np.append(all_bbs, bbs, axis=0)

        # Get the counts of each bb.
        tracked, counts = nms_counts(all_bbs, nms, counts=True)
        counts = np.array(counts)
        # Remove bbs with low counts.
        track_idx = np.where(counts >= min_count)[0]
        tracked = tracked[track_idx]

        # Convert sort output to sort xys
        sort_copy = self.sort_output.copy()
        sort_copy["w"] = (sort_copy["x2"] - sort_copy["x1"]) / 2
        sort_copy["h"] = (sort_copy["y2"] - sort_copy["y1"]) / 2

        sort_xy = np.transpose([sort_copy["x1"] + sort_copy["w"], sort_copy["y1"] + sort_copy["h"]])
        tracked_xy = tracked[:, :2]
        tracked_xy[:, 0] = tracked_xy[:, 0] + tracked[:, 2] / 2
        tracked_xy[:, 1] = tracked_xy[:, 1] + tracked[:, 3] / 2

        # Match df and the list of idxs from tracked bbs not found in sort.
        _, not_matched_idxs = find_best_matches(sort_xy, tracked_xy, min_distance=min_distance)
        to_track = tracked[not_matched_idxs]

        return to_track


if __name__ == "__main__":
    def test_load():
        sort_output = "data/results/4_v1/356.csv"
        yolo_csv = "data/samples/csvs/356.csv"
        video_path = "data/samples/vids/356.avi"
        end = 1300
        device = "cpu"

        obj = ReverseEncoder(sort_output, yolo_csv, video_path, end, device)
        print("Test Load -> Success")
        return obj

    def test_match():
        obj = test_load()
        to_track = obj.declare_woi()
        assert(len(to_track) == 5), "Error matching"
        print("Test Match -> Success")

    def test():
        test_match()

    test()