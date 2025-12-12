import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfiltfilt, lfilter, lfilter_zi, sosfilt_zi


class AudioPreprocessor:
    def __init__(self, sr=16000, hp_cutoff=100, notch_freq=50, notch_q=30):
        """
        Initializes filters based on PDF specifications.

        Args:
            sr (int): Sample rate (min 16kHz suggested by PDF Page 2).
            hp_cutoff (float): Cutoff for Highpass (80-100Hz suggested).
            notch_freq (float): Mains hum frequency (50Hz for EU, 60Hz for US).
            notch_q (float): Quality factor for Notch filter.
        """
        self.sr = sr
        self.nyquist = sr / 2

        # --- 1. Design High-Pass Filter (Butterworth) ---
        # Using SOS (Second-Order Sections) for better stability than 'ba'
        # Order 4 provides a sharp roll-off to kill low-freq noise
        self.sos_hp = butter(N=4, Wn=hp_cutoff / self.nyquist, btype='high', output='sos')

        # --- 2. Design Notch Filter (IIR Notch) ---
        # Removes specific mains hum (Page 3)
        w0 = notch_freq / self.nyquist
        self.b_notch, self.a_notch = iirnotch(w0, notch_q)

        # --- 3. Initialize States for Live Processing ---
        # These variables store the filter 'memory' between audio chunks
        self.zi_hp = sosfilt_zi(self.sos_hp)
        self.zi_notch = lfilter_zi(self.b_notch, self.a_notch)

    def process_offline(self, audio_data):
        """
        Best for file-based processing. Uses zero-phase filtering (filtfilt)
        to prevent phase distortion.
        """
        # 1. Normalization (Page 2: "Pegelnormalisierung")
        # Peak normalization to -1.0 to 1.0 range
        peak = np.max(np.abs(audio_data))
        if peak > 0:
            audio_data = audio_data / peak

        # 2. Apply High-Pass (Zero-phase)
        # sosfiltfilt is the SOS equivalent of filtfilt
        audio_hp = sosfiltfilt(self.sos_hp, audio_data)

        # 3. Apply Notch (Zero-phase)
        # Note: Scipy doesn't have a direct sosfiltfilt for b,a, so we use lfilter logic 
        # or convert notch to SOS. For simple notch, standard filtfilt with b,a is fine.
        from scipy.signal import filtfilt
        audio_clean = filtfilt(self.b_notch, self.a_notch, audio_hp)

        return audio_clean

    def process_live_chunk(self, chunk):
        """
        Best for continuous/streaming input. Maintains filter state.

        Args:
            chunk (np.array): A small block of audio samples (e.g., 1024 samples).

        Returns:
            np.array: The processed chunk.
        """
        # 1. Simple Safety Normalization / Clipping for Live
        # (Global peak normalization is impossible in live streams, 
        # so we ensure it doesn't exceed -1.0 to 1.0)
        chunk = np.clip(chunk, -1.0, 1.0)

        # 2. Apply High-Pass (Causal with State)
        # We pass the state (zi) in and get the updated state back
        chunk_hp, self.zi_hp = sosfilt(self.sos_hp, chunk, zi=self.zi_hp)

        # 3. Apply Notch (Causal with State)
        chunk_clean, self.zi_notch = lfilter(self.b_notch, self.a_notch, chunk_hp, zi=self.zi_notch)

        return chunk_clean

    def reset_states(self):
        """Call this if the audio stream restarts completely."""
        self.zi_hp = sosfilt_zi(self.sos_hp)
        self.zi_notch = lfilter_zi(self.b_notch, self.a_notch)


# ==========================================
# Example Usage
# ==========================================

if __name__ == "__main__":
    # Settings based on PDF
    SAMPLE_RATE = 16000

    # Initialize the processor
    processor = AudioPreprocessor(sr=SAMPLE_RATE, hp_cutoff=100, notch_freq=50)

    # --- Scenario A: Non-Live (File) ---
    # Simulate a full 3-second audio file
    full_audio = np.random.uniform(-0.5, 0.5, SAMPLE_RATE * 3)
    cleaned_file = processor.process_offline(full_audio)
    print(f"Offline Processed: {cleaned_file.shape}")

    # --- Scenario B: Live (Streaming) ---
    # Simulate streaming data in chunks of 1024 samples
    processor.reset_states()  # Clear memory before starting stream
    chunk_size = 1024

    print("Starting Live Stream Processing...")
    for i in range(5):  # Simulate 5 chunks coming in
        input_chunk = np.random.uniform(-0.5, 0.5, chunk_size)

        # This function remembers the end of the previous chunk
        output_chunk = processor.process_live_chunk(input_chunk)

        print(f"Chunk {i + 1}: Input RMS {np.std(input_chunk):.4f} -> Output RMS {np.std(output_chunk):.4f}")