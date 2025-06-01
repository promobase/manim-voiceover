import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from manim import logger

from manim_voiceover.helper import (
    prompt_ask_missing_extras,
    remove_bookmarks,
)
from manim_voiceover.services.base import SpeechService

try:
    import replicate
except ImportError:
    logger.error(
        "Missing packages. Run `pip install replicate` to use ReplicateService."
    )

load_dotenv(find_dotenv(usecwd=True))


class ReplicateService(SpeechService):
    """
    Speech service class for Replicate TTS Service.
    See https://replicate.com/minimax/speech-02-hd for more information.
    """

    def __init__(
        self,
        model: str = "minimax/speech-02-hd",
        voice_id: str = "Chinese (Mandarin)_Stubborn_Friend",
        pitch: int = 0,
        speed: float = 1.0,
        volume: float = 1.0,
        bitrate: int = 128000,
        channel: str = "mono",
        emotion: str = "happy",
        sample_rate: int = 32000,
        language_boost: str = "Chinese",
        english_normalization: bool = True,
        transcription_model="base",
        **kwargs,
    ):
        """
        Args:
            model (str): Replicate model name.
            voice_id (str): Voice ID to use.
            pitch (int): Pitch adjustment.
            speed (float): Speed adjustment.
            volume (float): Volume adjustment.
            bitrate (int): Bitrate in bps.
            channel (str): Audio channel.
            emotion (str): Emotion.
            sample_rate (int): Sample rate in Hz.
            language_boost (str): Language boost.
            english_normalization (bool): English normalization.
        """
        prompt_ask_missing_extras("replicate", "replicate", "ReplicateService")
        self.model = model
        self.voice_id = voice_id
        self.pitch = pitch
        self.speed = speed
        self.volume = volume
        self.bitrate = bitrate
        self.channel = channel
        self.emotion = emotion
        self.sample_rate = sample_rate
        self.language_boost = language_boost
        self.english_normalization = english_normalization

        SpeechService.__init__(self, transcription_model=transcription_model, **kwargs)

    def generate_from_text(
        self, text: str, cache_dir: str = None, path: str = None, **kwargs
    ) -> dict:
        if cache_dir is None:
            cache_dir = self.cache_dir

        # Allow overrides via kwargs
        pitch = kwargs.get("pitch", self.pitch)
        speed = kwargs.get("speed", self.speed)
        volume = kwargs.get("volume", self.volume)
        bitrate = kwargs.get("bitrate", self.bitrate)
        channel = kwargs.get("channel", self.channel)
        emotion = kwargs.get("emotion", self.emotion)
        voice_id = kwargs.get("voice_id", self.voice_id)
        sample_rate = kwargs.get("sample_rate", self.sample_rate)
        language_boost = kwargs.get("language_boost", self.language_boost)
        english_normalization = kwargs.get(
            "english_normalization", self.english_normalization
        )

        input_text = remove_bookmarks(text)
        input_data = {
            "input_text": input_text,
            "service": "replicate",
            "config": {
                "model": self.model,
                "voice_id": voice_id,
                "pitch": pitch,
                "speed": speed,
                "volume": volume,
                "bitrate": bitrate,
                "channel": channel,
                "emotion": emotion,
                "sample_rate": sample_rate,
                "language_boost": language_boost,
                "english_normalization": english_normalization,
            },
        }

        cached_result = self.get_cached_result(input_data, cache_dir)
        if cached_result is not None:
            return cached_result

        if path is None:
            audio_path = self.get_audio_basename(input_data) + ".mp3"
        else:
            audio_path = path

        if os.getenv("REPLICATE_API_TOKEN") is None:
            raise EnvironmentError(
                "REPLICATE_API_TOKEN environment variable is not set. "
            )

        output_obj = replicate.run(
            self.model,
            input={
                "text": input_text,
                "pitch": pitch,
                "speed": speed,
                "volume": volume,
                "bitrate": bitrate,
                "channel": channel,
                "emotion": emotion,
                "voice_id": voice_id,
                "sample_rate": sample_rate,
                "language_boost": language_boost,
                "english_normalization": english_normalization,
            },
            # Some replicate clients may require `output_as_dict=True`
        )

        # If output_obj is a dict, extract the audio URL from the "output" field
        if isinstance(output_obj, dict) and "output" in output_obj:
            audio_url = output_obj["output"]
        # If output_obj is a string (URL), use it directly (legacy behavior)
        elif isinstance(output_obj, str):
            audio_url = output_obj
        else:
            raise RuntimeError(
                "Unexpected output from replicate.run: {}".format(output_obj)
            )

        import requests

        response = requests.get(audio_url, stream=True)
        audio_file_path = str(Path(cache_dir) / audio_path)
        with open(audio_file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        json_dict = {
            "input_text": text,
            "input_data": input_data,
            "original_audio": audio_path,
        }

        return json_dict
