import cv2
import torch
import pandas as pd
import numpy as np

from classifier.classifier import WormClassifier
from utils.model_utils import ClassTransformer


class SortClassification:
    """
    Takes the sort outputs with death calls and frame location then uses
    the classifier to check if the worm moves out later. If the worm
    moves out over 144 frames then the death call is removed.
    """
    weights_path = "classifier/weights/316.pt"
    img_size = 24  # Input image size (img_size x img_size)
    transformer = ClassTransformer(img_size=img_size)
    scan_range = 200  # How far forward in time to scan the bounding box

    def __init__(self, sort_output, video_path: str, device="cpu"):
        # Define params.
        self.sort_output = sort_output  # String or pandas dataframe
        self.video_path = video_path
        self.device = device

        # Load Model.
        self.classifier = WormClassifier().to(self.device)
        self.classifier.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        print(f"Classifier -> Loaded weights from {self.weights_path} \nTo device: {self.device}")

        # Declare video and organize sort output.
        self.video = cv2.VideoCapture(self.video_path)
        self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

        if type(self.sort_output) == str:
            self.sort_output = pd.read_csv(self.sort_output)

        self.worm_ids = self.sort_output.label.unique()

    def classify(self, img):
        """Classifies the worms in the given img.
        Returns value between 0 and 1 predicting if worm is in the img or not."""
        model_input = self.transformer(img).to(self.device).unsqueeze(1)
        output = self.classifier(model_input)
        output = output.squeeze(0).cpu().detach().numpy()[0]
        return output

    def worm_from_frame(self, location, frame_id, pad=0):
        """Using worm_id gets the location from the sort_outputs
        and returns the croped image with the worm in it.

        Args:
            worm_id (int): Worm ID / label from sort_outputs.
            location (list[int]): [x1, y1, x2, y2]
            frame_id (int): Frame number.
        """
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
        ret, frame = self.video.read()  # Read frame
        x1, y1, x2, y2 = location

        # If pad is greater than frame size then pad is set to 0.
        # x1, x2 = x1 - pad, x2 + pad
        # y1, y2 = y1 - pad, y2 + pad
        worm = frame[y1:y2, x1:x2]
        return worm

    def process_worm(self, worm_id, window):
        """Takes worm id and then scans the video {144} frames ahead
        to see if the worm moves out using classifier.
        If it does then remove the death call.

        Args:
            worm_id (int): Worm ID / label from sort_outputs.
        """
        row = self.sort_output[self.sort_output.label == worm_id].iloc[0]
        loaction = [row.x1, row.y1, row.x2, row.y2]  # Worm location from sort.
        # Determine the lower and upper frame bounds to scan.
        lower = row.frame
        upper = lower + self.scan_range
        upper = upper if upper < self.frame_count else self.frame_count

        class_list = []
        # Scan the video.
        for frame_id in range(upper - window, upper):
            worm = self.worm_from_frame(loaction, frame_id)
            worm_class = self.classify(worm)
            class_list.append(worm_class)

        return class_list

    def process_all_worms(self, threshold=0.1):
        """Process all the worms from the sort output to check
        if they leave their designated bounding box."""
        updated_df = self.sort_output.copy()
        updated_df["real"] = True
        window = 20

        for worm_id in self.worm_ids:
            class_list = self.process_worm(worm_id, window)

            # class_list_smooth = savgol_filter(filter_size, 51, 2)
            # If the class is less than thresh the worm left the box.
            # if len(class_list) < 20:
                # window = -len(class_list) - 1

            # final_window = class_list[window:]
            # class_min = np.mean(final_window)
            class_min = np.mean(class_list)

            # class_min = np.min(final_window)
            # If there is no worm in the window in the last 20 frames then remove worm.
            if class_min < threshold:
                updated_df.loc[updated_df.label == worm_id, "real"] = False

        return updated_df

    def update_sort(self, threshold=0.1):
        """Updates the sort output with the new classification. Returns the updated sort output."""
        updated_df = self.process_all_worms(threshold=threshold)
        new_df = updated_df[updated_df.real]
        new_df = new_df.drop(columns=["real"])
        return new_df


if __name__ == "__main__":
    worm_id = 1000

    def test_load():
        sort = "data/results/samples/sort/356.csv"
        video_path = "data/samples/vids/356.avi"
        device = "cpu"

        obj = SortClassification(sort, video_path, device)
        print("Test Load Successful.")
        return obj

    def test_classify(obj):
        # worm_id = 10
        row = obj.sort_output[obj.sort_output.label == worm_id].iloc[0]
        loaction = [row.x1, row.y1, row.x2, row.y2]  # Worm location from sort.
        worm = obj.worm_from_frame(loaction, row.frame)
        worm_class = obj.classify(worm)
        assert(worm_class > 0.5), "Classification failed."

        location2 = [0, 0, 64, 64]
        worm2 = obj.worm_from_frame(location2, row.frame)
        worm_class2 = obj.classify(worm2)
        assert(worm_class2 < 0.5), "Classification failed."
        print("Test Classify Successful.")
        return worm_class

    def test_process_worm(obj):
        # worm_id = 10
        window = 20
        class_list = obj.process_worm(worm_id, window=window)
        assert(len(class_list) == window), "Processing failed."
        print("Test Process Worm Successful.")
        return class_list

    def test():
        obj = test_load()
        worm_class = test_classify(obj)
        class_list = test_process_worm(obj)
        return worm_class, class_list

    a = test()
