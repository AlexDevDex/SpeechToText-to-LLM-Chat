import asyncio
import aiohttp
import aiofiles
from ollama import AsyncClient
import pyaudio
import wave
import os
import numpy as np

# Set ollama server address
client = AsyncClient(host='http://192.168.1.42:11434')

# Set the whisper server URL and headers
url = "http://192.168.1.42:9000/asr"
headers = {"accept": "application/json"}

# Set the audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 4096

# Initialize VAD parameters
ENERGY_THRESHOLD = 1000  # Adjust this threshold based on your environment
SILENCE_FRAMES = 6       # Number of frames to wait before stopping recording after silence

def vad(frame):
    energy = np.sum(np.frombuffer(frame, dtype=np.int16)**2)
    return energy > ENERGY_THRESHOLD

def record_vad(p, stream, file_path):
    frames = []
    silence_count = 0

    while True:
        data = stream.read(CHUNK)
        frames.append(data)

        if vad(data):
            silence_count = 0
        else:
            silence_count += 1

        if silence_count >= SILENCE_FRAMES:
            break

    wf = wave.open(file_path, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return file_path  # Return the file path for further use

async def chat(session, response):
    message = {'role': 'user', 'content': response.text}

    async with session.post(url, headers=headers, data=message) as resp:
        if resp.status == 200:
            async for part in await resp.content:
                print(part['message']['content'], end='', flush=True)
        else:
            print(f"Error: {resp.status}")

async def primary():
    async with aiohttp.ClientSession() as session:
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

        transcription_log = ""

        try:
            while True:
                chunk_file = "temp_chunk.wav"
                chunk_file = record_vad(p, stream, chunk_file)  # Update to get the file path
                params = {
                    "encode": "true",
                    "task": "transcribe",
                    "language": "de"
                }
                async with aiofiles.open(chunk_file, "rb") as file:
                    async with session.post(url, data={"audio_file": file}, params=params) as resp:
                        if resp.status == 200:
                            response_text = await resp.text()
                            if response_text.strip() not in ["Vielen Dank für's Zuschauen.", "Vielen Dank für's Zuschauen!", "SWR 2020", "Vielen Dank."]:
                                transcription = response_text
                                print(transcription)
                        else:
                            print(f"Error: {resp.status}")

                os.remove(chunk_file)
                transcription_log += transcription + " "
        
        except KeyboardInterrupt:
            print("Stopping...")
            async with aiofiles.open("log.txt", "w") as log_file:
                await log_file.write(transcription_log)

        finally:
            print("Log:" + transcription_log)
            stream.stop_stream()
            stream.close()
            p.terminate()

if __name__ == "__main__":
    asyncio.run(primary())
