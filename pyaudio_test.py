import wave
import time
import sys

import pyaudio




# Instantiate PyAudio and initialize PortAudio system resources (2)
p = pyaudio.PyAudio()

# Open stream using callback (3)
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024)

accumulated_transcript = ""

try:
    while True:
        chunk_file = "temp_chunk.wav"
        record_chunk(p, stream, chunk_file)

stream.close()
p.terminate()