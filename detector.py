import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import csv

class ScreamDetector:
    def __init__(self, sensitivity=0.4):
        """
        Initializes the YAMNet Deep Learning model.
        Args:
            sensitivity (float): 0.0 to 1.0. Lower = detects faint screams (more false positives).
        """
        self.sensitivity = sensitivity

        print("Loading AI Model (YAMNet)... this may take a moment...")
        # Load Google's pre-trained audio event model
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')

        # Load the class names (YAMNet identifies 521 sounds, we only want screams)
        class_map_path = self.model.class_map_path().numpy()
        self.class_names = self._load_class_names(class_map_path)

        # Find the specific indices for "Screaming" and "Yelling"
        self.target_indices = []
        for i, name in enumerate(self.class_names):
            if name in ['Screaming', 'Yell', 'Shout']:
                self.target_indices.append(i)

        # Buffer settings
        # YAMNet expects ~0.975 seconds of audio at 16khz
        self.sr = 16000
        self.buffer_size = 15600  # 0.975s * 16000
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)

    def _load_class_names(self, csv_path):
        """Helper to parse the class map CSV from the model."""
        class_names = []
        with tf.io.gfile.GFile(csv_path) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                class_names.append(row['display_name'])
        return class_names

    def process_live_stream(self, chunk):
        """
        Feeds a small chunk (e.g., 1024 samples) into the rolling buffer
        and runs detection.

        Returns:
            (bool, float, str): (Detected?, Confidence Score, Label)
        """
        chunk_len = len(chunk)

        # 1. Rolling Buffer: Shift old data left, add new data to right
        self.buffer = np.roll(self.buffer, -chunk_len)
        self.buffer[-chunk_len:] = chunk

        # 2. Run Inference
        # Note: In a production system, you might not want to run this
        # on *every* chunk if the chunks are very small (high CPU load).
        # You could add a counter to run it every 4th chunk.
        scores, embeddings, spectrogram = self.model(self.buffer)

        # scores is shape (N, 521). We take the mean across the short window.
        prediction = np.mean(scores.numpy(), axis=0)

        # 3. Check for Screams
        # We sum the probabilities of "Scream", "Yell", "Shout"
        scream_score = np.sum(prediction[self.target_indices])

        # Find the dominant sound (for debugging)
        top_class_index = np.argmax(prediction)
        top_class_name = self.class_names[top_class_index]

        if scream_score > self.sensitivity:
            return True, scream_score, top_class_name

        return False, scream_score, top_class_name


