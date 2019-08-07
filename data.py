import pandas as pd
from pydub import AudioSegment
import numpy as np

class Data:
    filename = None
    __index = 0

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.load_csv()

    def load_csv(self):
        df = pd.read_csv(self.csv_path)
        self.filename = df['filename']

    def load_sound_sample(self, filename):
        sound = AudioSegment.from_file(filename, 'mp3')
        samples = np.array(sound.get_array_of_samples())
        return np.float32(samples) / 2**15 # Normalize [-1, 1)

    def make_chunk(self, array, num_samples):
        for i in range(len(array) % num_samples): array = np.delete(array, 0)
        num_chunks = len(array) / num_samples
        chunk = np.split(array, num_chunks)
        return chunk

    def make_batch(self, num_audiofile):
        batch = np.empty((0, 512))
        for i in range(num_audiofile):
            samples = self.load_sound_sample(self.filename[self.__index])
            chunks = self.make_chunk(samples, 512)
            batch = np.append(batch, chunks, axis = 0)
            self.__index = self.__index + 1
        return batch

    def add_noise(self, raw, mean, dev):
        noise = np.random.normal(mean, dev, raw.shape)
        return np.add(raw, noise)
