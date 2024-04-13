import asyncio
from ollama import AsyncClient
import pyaudio
import wave
import os
import numpy as np
import httpx

# Set ollama server settings
client = AsyncClient(host='http://192.168.1.42:11434')
MODEL = "mayflowergmbh/wiedervereinigung"

# Set whisper server settings
url = "http://192.168.1.42:9000/asr"
LANG = "de"

# Set the audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 4096

# Initialize VAD parameters
ENERGY_THRESHOLD = 1000  # Adjust this threshold based on your environment
SILENCE_FRAMES = 6       # Number of frames to wait before stopping recording after silence

# Calculate if there is sound to record
def vad(frame):
    energy = np.sum(np.frombuffer(frame, dtype=np.int16)**2)
    #print(energy > ENERGY_THRESHOLD)
    return energy > ENERGY_THRESHOLD

# Record sound/voice until it is silent again
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

# TODO save the answer
# Send the transcribed text from whisper to ollama
async def chat_with_ollama(text):
    message = {'role': 'user', 'content': text}
    async for part in await client.chat(model=MODEL, messages=[message], stream=True, options={
        #"num_keep": 5,
        #"seed": 42, # Sets the random number seed to use for generation, ensuring reproducibility. The default value is 42.
        #"num_predict": 100, # Specifies the maximum number of tokens to predict when generating text. The default value is 42.
        #"top_k": 20, # Reduces the probability of generating nonsense. Higher values give more diverse answers. The default value is 40.
        #"top_p": 0.9, # Works together with top-k. Higher values lead to more diverse text. The default value is 0.9.
        #"tfs_z": 0.5, # Tail free sampling parameter. Higher values reduce the impact of less probable tokens. The default value is 1.
        #"typical_p": 0.7,
        #"repeat_last_n": 64, # Sets how far back the model looks to prevent repetition. The default value is 64.
        #"temperature": 0.8, # Controls the "creativity" of the model's responses. Higher values lead to more creative responses. The default value is 0.7.
        #"repeat_penalty": 1.2, # Sets how strongly to penalize repetitions. Higher values penalize repetitions more strongly. The default value is 1.1.
        #"presence_penalty": 1.5,
        #"frequency_penalty": 1.0,
        #"mirostat": 1, # This option enables Mirostat sampling for controlling perplexity. It has three possible values: 0 (disabled), 1 (Mirostat), and 2 (Mirostat 2.0).
        #"mirostat_tau": 0.8, # Controls the balance between coherence and diversity of the output. A lower value results in more focused and coherent text. The default value is 5.0.
        #"mirostat_eta": 0.6, # This parameter influences how quickly the algorithm responds to feedback from the generated text. A lower learning rate results in slower adjustments, while a higher learning rate makes the algorithm more responsive. The default value is 0.1.
        #"penalize_newline": False,
        #"stop": ["\n", "user:"], # Specifies the stop sequences to use. When encountered, the model stops generating text and returns. The default stop sequence is "AI assistant:"
        #"numa": False,
        #"num_ctx": 4096, # Sets the size of the context window used to generate the next token. The default value is 4096.
        #"num_batch": 2,
        #"num_gqa": 1, # Specifies the number of GQA (Generative Query Answering) groups in the transformer layer. The default value is 1.
        #"num_gpu": 50, # Specifies the number of layers to send to the GPU(s). The default value is 50
        #"main_gpu": 0,
        #"low_vram": False,
        #"f16_kv": True,
        #"vocab_only": False,
        #"use_mmap": True,
        #"use_mlock": False,
        #"rope_frequency_base": 1.1,
        #"rope_frequency_scale": 0.8,
        #"num_thread": 8 # Sets the number of threads to use during computation. The default value is 8.
        }
    ):
        print(part['message']['content'], end='', flush=True)

# Main asynchronous function to handle audio recording and text generation
async def primary():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    transcription_log = ""

    try:
        async with httpx.AsyncClient() as client:
            while True:
                chunk_file = "temp_chunk.wav"
                chunk_file = record_vad(p, stream, chunk_file)  # Update to get the file path
                
                files = {"audio_file": open(chunk_file, "rb")}

                params = {
                    "encode": "true",
                    "task": "transcribe",
                    "language": LANG
                }

                response = await client.post(url, headers={}, files=files, params=params)
                files["audio_file"].close()
                transcription = ""  # Default value

                if response.status_code == 200:
                    response_text = response.text.strip()
                    if response_text not in ["Vielen Dank für's Zuschauen.", "Vielen Dank für's Zuschauen!", "SWR 2020", "Vielen Dank.", "Untertitel im Auftrag des ZDF für funk, 2017"]:
                        transcription = response.text
                        if len(transcription) > 1:
                            asyncio.create_task(chat_with_ollama(transcription))
                            print(transcription)
                else:
                    print(response.json())
                
                os.remove(chunk_file)
                transcription_log += transcription + " "
        
    except KeyboardInterrupt:
        print("Stopping gracefully...")
        with open("log.txt", "w") as log_file:
            log_file.write(transcription_log)
        
    except asyncio.CancelledError:
        print("Async task cancelled")

    finally:
        print("Log:" + transcription_log)
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    asyncio.run(primary())
