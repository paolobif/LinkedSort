import torch
import cv2
import pandas as pd
import numpy as np

from tqdm import tqdm
from scipy.signal import savgol_filter


from autoencoder.autoencoder import Autoencoder
from utils.model_utils import Transformer
from utils.utils import nms_counts, screen_exp, first_death, last_death
from utils.encode_utils import *


class ReverseEncoder():
    """Gets worms of interest from yolo detections at the frame specified by sort.
    Also gets the outputs from the classification. For any missed calls it'll run
    back with the autoencoder to updated the list of worm calls.
    """
    weights_path = "autoencoder/weights/autoencoder_4.pt"
    img_size = 28
    transformer = Transformer(thresh_use=True, square_use=True, img_size=img_size)

    def __init__(self, sort_output, yolo_csv: str, video_path: str, bounds: tuple = False, device="cpu"):
        """
        Args:
            sort_output (pd DF | str): Analyze sort output.
            yolo_csv (str): Path to yolo detections.
            video_path (str): Path to video avi.
            bounds (tuple, optional): Bounds of experiment (0, 1000). Defaults to False.
            device (str, optional): Torch device. Defaults to "cpu".
        """
        self.sort_output = sort_output
        if type(self.sort_output) == str:
            self.sort_output = pd.read_csv(self.sort_output)

        self.expID = self.sort_output["expID"].iloc[0]
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

        # Get experiment bounds. [start, stop]
        if bounds:
            self.bounds = bounds
        else:
            print("Getting experiment bounds...")
            self.bounds = self.experiment_bounds()

        # Worms of interest that are not picked up by sort.
        self.woi = self.declare_woi(window=10, min_count=2, nms=0.7)

    def experiment_bounds(self, pad: int = 75, val: float = 0):
        """ Gets the upper and lower bounds of an experiment based
        on their worm entropy.

        Args:
            pad (int, optional): Add pad frames to the determined end. Defaults to 0.
            val (int, optional): Slope at wich exp end is determined. Defaults to 0.
        """
        interval = 5  # Frame count of moving screen window.
        counts = np.array(screen_exp(self.df, interval, self.expID))
        frames = counts[:, 1]
        y = counts[:, 2]
        y_deriv = savgol_filter(y, 101, 2, deriv=1)
        exp_end = last_death(y_deriv, frames, val=val, pad=pad)
        exp_start = first_death(y_deriv, frames, val=0.6)  # 0.6 is slope where worms start dying.
        return exp_start, exp_end

    def get_frame(self, frame_id):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
        ret, frame = self.video.read()  # Read frame
        return frame

    def get_bbs_from_frame(self, frame_id):
        """ Gets the frame image from a frame id,
        and then the bounding boxes associated with that image"""
        frame = self.get_frame(frame_id)
        bbs = self.df[self.df["frame"] == frame_id]
        bbs = bbs.to_numpy()
        bbs = bbs[:, 1:5]
        return frame, bbs

    def get_worm_from_frame(self, bb: list, frame_id: int, pad=(0, 0)):
        """ Gets the worm from a bounding box and frame. Apply
        padding if specified.

        Args:
            bb (list): Bounding box of worm [x, y, w, h].
            frame_id (int): Frame id to fetch from.
            pad (tuple, optional): Pad (x, y). Defaults to (0, 0).

        Returns:
            np.ndarray: Worm image.
        """
        assert(frame_id < self.frame_count), "Frame id out of bounds."
        frame = self.get_frame(frame_id)
        size_x, size_y, _ = frame.shape
        x, y, w, h = bb
        x1, x2 = x - pad[0], x + w + pad[0]
        y1, y2 = y - pad[1], y + h + pad[1]
        x1, x2 = max(0, x1), min(size_x, x2)
        y1, y2 = max(0, y1), min(size_y, y2)
        worm = frame[y1:y2, x1:x2]
        return worm

    def declare_woi(self, window=10, min_count=2, nms=0.7):
        """Declares worms of interest at the point where sort makes the last
        time of death call. Compares this to already called points by sort
        and determines which worms need to be tracked by encoder.
        """
        min_distance = 40  # Min distance to match sort to new woi.
        end = self.bounds[1]
        first = end - window
        last = end

        all_bbs = np.empty((0, 4), int)
        for i in range(first, last):
            _, bbs = self.get_bbs_from_frame(i)
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
        to_track = to_track.astype(int)

        return to_track

    def encode_img(self, img):
        input_img = self.transformer(img)
        with torch.no_grad():
            input_img = input_img.to(self.device)
            input_img = input_img.unsqueeze(1)
            output = self.encoder.forward(input_img, encode=True)
            output = output.squeeze(0).detach().cpu()

        return output

    def process_worm(self, bb: list, lower: int, upper: int):
        """ Processes a worm though the encoder. Then compare the output
        vector to the start and end of the experiment.
        Args:
            bb (np array): Bounding box of worm [x, y, w, h].
            lower (int): Lower bound of frame id.
            upper (int): Upper bound of frame id.
        """
        skip = 5  # Skip every n frames.
        ref_lower = self.encode_img(self.get_worm_from_frame(bb, lower, pad=(2, 2)))
        ref_upper = self.encode_img(self.get_worm_from_frame(bb, upper, pad=(2, 2)))

        distances = []  # Distance from start and end of exp.
        frames = []  # Tracks the frame id of each distance.

        for frame_id in range(lower, upper, skip):
            # Get the worm from the frame.
            worm = self.get_worm_from_frame(bb, frame_id, pad=(2, 2))
            output = self.encode_img(worm)
            upper_dist = get_distance(output, ref_upper)
            lower_dist = get_distance(output, ref_lower)
            distances.append([lower_dist, upper_dist])
            frames.append(frame_id)

        # Smooth the distances.
        distances = np.array(distances)  # [lower, upper] respective distances.
        distances[:, 0] = savgol_filter(distances[:, 0], 21, 3)
        distances[:, 1] = savgol_filter(distances[:, 1], 21, 3)

        encoded_tod = get_worm_tod(distances, frames)
        return encoded_tod

    def process_wois(self, save_path=None):
        """Processes the worms of interest through the encoder."""
        output = self.sort_output.copy()
        output = output.drop(columns=["label"])

        upper, lower = self.bounds
        print("Bulk encoding wois.")
        for bb in tqdm(self.woi):
            encoded_tod = self.process_worm(bb, upper, lower)
            x1, y1, w, h = bb
            x2, y2 = x1 + w, y1 + h
            row = {"frame": encoded_tod, "x1": x1, "y1": y1,
                   "x2": x2, "y2": y2, "expID": self.expID}
            output = output.append(row, ignore_index=True)

        if save_path:
            output.to_csv(save_path, index=False)

        return output


if __name__ == "__main__":
    bulk_test = True

    def test_load():
        sort_output = "data/results/4_v1/356.csv"
        yolo_csv = "data/samples/csvs/356.csv"
        video_path = "data/samples/vids/356.avi"
        device = "cpu"

        obj = ReverseEncoder(sort_output, yolo_csv, video_path, device=device)
        print("Test Load -> Success")
        assert(obj.bounds == (357, 1441)), "Experiment bounds incorrect"
        print("Test Bounds -> Success")
        return obj

    def test_match():
        obj = test_load()
        to_track = obj.declare_woi()
        # print(to_track)
        assert(len(to_track) == 4), "Error matching"
        print("Test Match -> Success")
        return obj

    def test_encode():
        obj = test_match()
        bb = obj.woi[2]
        pad = (2, 2)
        worm = obj.get_worm_from_frame(bb, obj.bounds[1], pad=pad)
        assert(worm.shape[0] == bb[3] + 2 * pad[1]), "Worm shape incorrect"
        assert(worm.shape[1] == bb[2] + 2 * pad[0]), "Worm shape incorrect"
        encoded_worm = obj.encode_img(worm)
        assert(encoded_worm.shape[0] == 64), "Encoded shape incorrect"
        print("Test Encode -> Success")
        return obj

    def test_process():
        save_path = "data/results/samples/356_auto.csv"
        obj = test_encode()
        bb = obj.woi[3]
        lower, upper = obj.bounds
        obj.process_worm(bb, lower, upper)
        print("Test Single Process -> Success")
        if bulk_test:
            output = obj.process_wois(save_path)
            print("Test Bulk Process -> Success")
            print(output)
        else:
            print("Bulk test disabled")

    def test():
        test_process()

    test()
