import asyncio
from ollama import AsyncClient
import requests

client = AsyncClient(host='http://192.168.1.42:11434')

# audio file path
audio_file_path = "recording.wav"

url = "http://192.168.1.42:9000/asr"

headers = {
    "accept": "application/json"
}

files = {
    "audio_file": open(audio_file_path, "rb")
}

params = {
    "encode": "true",
    "task": "transcribe",
    "language": "en"
}

response = requests.post(url, headers=headers, files=files, params=params)

if response.status_code == 200:
    #print("worked")
    print(response.text)
else:
    print(response.json())


async def chat():
  message = {'role': 'user', 'content': response.text}
  async for part in await client.chat(model='mistral', messages=[message], stream=True):
    print(part['message']['content'], end='', flush=True)

asyncio.run(chat())