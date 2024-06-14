import wave, struct, os
from openai import OpenAI
from pvrecorder import PvRecorder
from playsound import playsound
from IPython.display import Image, display

 # silence deprecation warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

client = OpenAI(api_key="<Your API Key>")

class Chatbot:
    def __init__(self, client):
        self.client = client
        self.context = [
            {"role": "system", "content": "You are a witty assistant, always answering with a Joke."},
        ]

    def chat(self, message):
        self.context.append(
            {"role": "user", "content": message}
        )
        response = self.client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=self.context
        )
        response_content = response.choices[0].message.content
        self.context.append(
            {"role": "assistant", "content": response_content}
        )
        self.show_face(response_content)
        self.print_chat()
        self.speak(response_content)

    def speak(self, message, index=0):
        speech_file_path = os.getcwd() + f"/speech_{index}.mp3"
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice="echo",
            input=message
        )
        response.stream_to_file(speech_file_path)
        playsound(speech_file_path)

    def record_audio(self, index=0):
        recorder = PvRecorder(device_index=-1, frame_length=512)
        audio = []
        filepath = os.getcwd() + f"/recorded_{index}.mp3"
        
        try:
            recorder.start()
            print("Audio recording started ...")
            while True:
                frame = recorder.read()
                audio.extend(frame)
        except KeyboardInterrupt:
            recorder.stop()
            print("Audio recording stopped ...")
            with wave.open(filepath, 'w') as f:
                f.setparams((1, 2, 16000, 512, "NONE", "NONE"))
                f.writeframes(struct.pack("h" * len(audio), *audio))
        finally:
            recorder.delete()
            return filepath

    def transcribe(self, audio_path):
        audio_file= open(audio_path, "rb")
        transcript = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )
        return transcript.text 

    def voicechat(self):
        recorded_filepath = self.record_audio(index=len(self.context))
        message = self.transcribe(recorded_filepath)
        self.chat(message)

    def show_face(self, message):
        response = self.client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": """
                    You are a face describing system which describes the face of a funny person making a funny comment.
						You receive the text the person is saying, please describe the face that would be fitting as a prompt to the stable diffusion image generation AI, DALL·E.
                """},
                {"role": "user", "content": message},
              ]
        )
        image_description = response.choices[0].message.content

        response = client.images.generate(
            model="dall-e-3",
            prompt=image_description,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        
        display(Image(url=response.data[0].url))

    def print_chat(self):
        for message in self.context:
            if message["role"] == "user":
                print(f'USER: {message["content"]}')
            elif message["role"] == "assistant":
                print(f'BOT: {message["content"]}')


chatbot = Chatbot(client)

chatbot.voicechat()