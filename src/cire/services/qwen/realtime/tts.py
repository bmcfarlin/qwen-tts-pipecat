"""
Qwen TTS Service for Pipecat
Date: Sept 6, 2026
Version: 1.1.2
Pipecat Version: 0.0.101
Author: Ben McFarlin <ben@qeala.com>
GitHub: https://github.com/bmcfarlin
Qwen Code AI Agent: https://qwen.ai/qwencode
User Guide:    https://www.alibabacloud.com/help/en/model-studio/speech-synthesis
Voice List:    https://www.alibabacloud.com/help/en/model-studio/qwen-tts-voice-list 
API Reference: https://www.alibabacloud.com/help/en/model-studio/qwen-tts-realtime-api-reference
Language Support: Chinese (Mandarin), English, German, Italian, Portuguese, Spanish, Japanese, Korean, French, and Russian
Note: This version support qwen3-tts-instruct-flash-realtime, qwen3-tts-flash-realtime, and qwen-tts-realtime. Support for qwen-audio is planned for the future.
"""


import json
import asyncio
from loguru import logger
import base64
from typing import AsyncGenerator, Optional
import string
import secrets

from pipecat.processors.frame_processor import FrameDirection
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
)
from pipecat.utils.text.base_text_aggregator import BaseTextAggregator
from pipecat.services.tts_service import WebsocketTTSService
from websockets.asyncio.client import connect as websocket_connect
from websockets.protocol import State
from pipecat.utils.tracing.service_decorators import traced_tts
from websockets.exceptions import ConnectionClosed


class ModelType:
    """Model type constants for Qwen TTS API."""

    QWEN3_TTS_INSTRUCT_FLASH_REALTIME = "qwen3-tts-instruct-flash-realtime"
    QWEN3_TTS_FLASH_REALTIME = "qwen3-tts-flash-realtime"
    QWEN_TTS_REALTIME = "qwen-tts-realtime"


class AudioFormat:
    """Audio format constants for Qwen TTS API."""

    PCM = "pcm"
    WAV = "wav"
    MP3 = "mp3"
    OPUS = "opus"


class InteractionMode:
    """Interaction mode constants for Qwen TTS API."""

    SERVER_COMMIT = "server_commit"
    COMMIT = "commit"


class QwenTTSService(WebsocketTTSService):
    """Qwen TTS service using DashScope Realtime WebSocket API.

    Uses commit mode: text is buffered via input_text_buffer.append,
    then synthesized on input_text_buffer.commit (sent in flush_audio).
    """

    def __init__(
        self,
        *,
        api_key: str,
        voice: str = "Cherry",
        model: ModelType = ModelType.QWEN3_TTS_INSTRUCT_FLASH_REALTIME,
        url: str = "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime",
        mode: InteractionMode = InteractionMode.COMMIT,
        language_type: Optional[str] = None,
        response_format: AudioFormat = AudioFormat.PCM,
        sample_rate: Optional[int] = None,
        speech_rate: Optional[float] = None,
        volume: Optional[int] = None,
        pitch_rate: Optional[float] = None,
        bit_rate: Optional[int] = None,
        instructions: Optional[str] = None,
        optimize_instructions: Optional[bool] = None,
        text_aggregator: Optional[BaseTextAggregator] = None,
        aggregate_sentences: Optional[bool] = True,
        user_agent: Optional[str] = None,
        workspace_id: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the Qwen TTS service.

        Args:
            api_key: DashScope API key for authentication.
            voice: Voice name for synthesis.
            model: TTS model identifier.
            url: WebSocket URL for DashScope Realtime API.
            mode: Interaction mode (server_commit or commit).
            language_type: Language of synthesized text (Auto, Chinese, English, etc.).
            response_format: Audio output format (pcm, wav, mp3, opus).
            sample_rate: Audio sample rate in Hz (8000, 16000, 24000, 48000).
            speech_rate: Playback speed (0.5-2.0). Not supported by Qwen-TTS-Realtime.
            volume: Audio volume (0-100). Not supported by Qwen-TTS-Realtime.
            pitch_rate: Audio pitch (0.5-2.0). Not supported by Qwen-TTS-Realtime.
            bit_rate: Audio bitrate in kbps (6-510, opus only). Not supported by Qwen-TTS-Realtime.
            instructions: Style and expressiveness control (max 1600 tokens). Instruct model only.
            optimize_instructions: Rewrite instructions for better naturalness. Instruct model only.
            text_aggregator: Custom text aggregator.
            aggregate_sentences: Whether to aggregate sentences.
            user_agent: Client identifier for server-side source tracking.
            workspace_id: DashScope workspace ID.
            **kwargs: Additional arguments passed to parent service.
        """
        super().__init__(
            aggregate_sentences=aggregate_sentences,
            pause_frame_processing=True,
            push_stop_frames=True,
            sample_rate=sample_rate,
            text_aggregator=text_aggregator,
            **kwargs,
        )
        self._api_key = api_key
        self._voice = voice
        self._model = model
        self._url = url
        self._mode = mode
        self._language_type = language_type
        self._response_format = response_format
        self._speech_rate = speech_rate
        self._volume = volume
        self._pitch_rate = pitch_rate
        self._bit_rate = bit_rate
        self._instructions = instructions
        self._optimize_instructions = optimize_instructions
        self._user_agent = user_agent
        self._workspace_id = workspace_id

        self._websocket = None
        self._receive_task = None
        self._session_updated = asyncio.Event()
        self._ttfb_started = False

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate metrics.

        Returns:
            True, as this service supports metrics generation.
        """
        return True

    async def start(self, frame: StartFrame):
        """Start the Qwen TTS service.

        Args:
            frame: The start frame.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Qwen TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Qwen TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with flush on LLM response end.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, (LLMFullResponseEndFrame, EndFrame)):
            await self.flush_audio()

    async def flush_audio(self):
        """Send commit to trigger synthesis of buffered text."""
        if self._websocket and self._websocket.state is State.OPEN:
            msg = self._build_commit_msg()
            logger.debug(f"flush_audio: {msg}")
            await self._websocket.send(msg)

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Buffer text for synthesis.

        Sends text to the server's input buffer via input_text_buffer.append.
        The actual synthesis is triggered by flush_audio (commit).

        Args:
            text: The text to synthesize.

        Yields:
            Frame: TTSStartedFrame on first call, then None.
        """
        logger.debug(f"Generating TTS [{text}]")

        try:
            if not self._session_updated.is_set():
                await self._session_updated.wait()

            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            try:
                msg = self._build_msg(text=text)
                logger.debug(msg)
                await self._websocket.send(msg)
                await self.start_tts_usage_metrics(text)
                if not self._ttfb_started:
                    await self.start_ttfb_metrics()
                    self._ttfb_started = True
                yield TTSStartedFrame()
            except Exception as e:
                yield ErrorFrame(error=f"Unknown error occurred: {e}")
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")

    async def _receive_messages(self):
        """Process incoming WebSocket messages from DashScope."""
        try:
            async for message in self._websocket:
                data = json.loads(message)
                event_type = data.get("type")
                logger.debug(f"event_type: {event_type}")

                if event_type == "session.created":
                    session_msg = self._build_session_update_msg()
                    logger.debug(session_msg)
                    await self._websocket.send(session_msg)

                elif event_type == "session.updated":
                    self._session_updated.set()
                    logger.debug(data)

                elif event_type == "response.audio.delta":
                    audio_b64 = data.get("delta")
                    if audio_b64:
                        await self.stop_ttfb_metrics()
                        self._ttfb_started = False
                        pcm_bytes = base64.b64decode(audio_b64)
                        frame = TTSAudioRawFrame(
                            audio=pcm_bytes,
                            sample_rate=self._sample_rate,
                            num_channels=1,
                        )
                        await self.push_frame(frame)

                elif event_type == "response.done":
                    logger.debug("response.done")

                elif event_type == "error":
                    error_obj = data.get("error", {})
                    error_code = error_obj.get("code", "unknown")
                    error_message = error_obj.get("message", "Unknown error")
                    logger.error(f"DashScope server error [{error_code}]: {error_message}")
                    await self.push_error(
                        error_msg=f"DashScope error [{error_code}]: {error_message}"
                    )

        except ConnectionClosed as e:
            logger.warning(f"websocket closed: code={e.code}, reason={e.reason}")
        except asyncio.CancelledError:
            logger.debug("_receive_messages task cancelled")
        except Exception as e:
            logger.exception(f"Unexpected error in _receive_messages: {e}")

    async def _connect(self):
        """Connect to DashScope and start receive task."""
        await super()._connect()
        await self._connect_websocket()
        if not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Disconnect from DashScope and clean up tasks."""
        await super()._disconnect()
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Establish or re-establish the WebSocket connection."""
        try:
            if self._websocket:
                if self._websocket.state is State.CLOSED:
                    await self._websocket_connect()
            else:
                await self._websocket_connect()
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close the WebSocket connection and reset state."""
        try:
            if self._websocket:
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            self._session_updated.clear()
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    async def _websocket_connect(self):
        """Open a new WebSocket connection to DashScope."""
        url = f"{self._url}?model={self._model}"
        additional_headers = {"Authorization": f"Bearer {self._api_key}"}
        if self._user_agent:
            additional_headers["user-agent"] = self._user_agent
        if self._workspace_id:
            additional_headers["X-DashScope-WorkSpace"] = self._workspace_id
        logger.debug("websocket_connect")
        self._websocket = await websocket_connect(url, additional_headers=additional_headers)
        await self._call_event_handler("on_connected")

    def _generate_unique_id(self, prefix: str = None, length: int = 22) -> str:
        """Generate a random identifier string.

        Args:
            prefix: Optional prefix for the ID.
            length: Length of the random portion.

        Returns:
            A unique identifier string.
        """
        charset = string.ascii_letters + string.digits
        result = "".join(secrets.choice(charset) for _ in range(length))
        if prefix:
            result = f"{prefix}_{result}"
        return result

    def _build_session_update_msg(self) -> str:
        """Build the session.update message.

        Returns:
            JSON string for the session update.
        """
        session = {
            "voice": self._voice,
            "mode": self._mode,
            "response_format": self._response_format,
        }
        if self._sample_rate:
            session["sample_rate"] = self._sample_rate
        if self._language_type:
            session["language_type"] = self._language_type
        if self._speech_rate is not None:
            session["speech_rate"] = self._speech_rate
        if self._volume is not None:
            session["volume"] = self._volume
        if self._pitch_rate is not None:
            session["pitch_rate"] = self._pitch_rate
        if self._bit_rate is not None:
            session["bit_rate"] = self._bit_rate
        if self._instructions:
            session["instructions"] = self._instructions
        if self._optimize_instructions is not None:
            session["optimize_instructions"] = self._optimize_instructions

        return json.dumps({
            "event_id": self._generate_unique_id("event"),
            "type": "session.update",
            "session": session,
        })

    def _build_msg(self, text: str = "") -> str:
        """Build an input_text_buffer.append message.

        Args:
            text: The text to append to the buffer.

        Returns:
            JSON string for the append message.
        """
        return json.dumps({
            "event_id": self._generate_unique_id("event"),
            "type": "input_text_buffer.append",
            "text": text,
        })

    def _build_commit_msg(self) -> str:
        """Build an input_text_buffer.commit message.

        Returns:
            JSON string for the commit message.
        """
        return json.dumps({
            "event_id": self._generate_unique_id("event"),
            "type": "input_text_buffer.commit",
        })

    def _build_clear_msg(self) -> str:
        """Build an input_text_buffer.clear message.

        Returns:
            JSON string for the clear message.
        """
        return json.dumps({
            "event_id": self._generate_unique_id("event"),
            "type": "input_text_buffer.clear",
        })

    def _build_finish_msg(self) -> str:
        """Build a session.finish message.

        Returns:
            JSON string for the finish message.
        """
        return json.dumps({
            "event_id": self._generate_unique_id("event"),
            "type": "session.finish",
        })
