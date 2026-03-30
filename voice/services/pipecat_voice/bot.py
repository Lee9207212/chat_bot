"""
bot.py — Pipecat Voice Pipeline 核心

組裝 Pipeline: WebRTC Input → VAD → STT → WebRTC Output
支援 STT 雙模式（本地 GPU / 遠端 HTTP）。
"""

import logging

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from config import config

logger = logging.getLogger("pipecat-voice-bot")


async def run_bot(webrtc_connection):
    """
    為每個 WebRTC 連線建立一個獨立的 Pipecat Pipeline。
    由 server.py 的 webrtc_connection_callback 呼叫。
    """
    logger.info("Starting new voice bot pipeline")

    # ── Transport ─────────────────────────────────────────
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            # Voice 作為「輸入層」時，不由 Pipecat 直接回播音訊。
            audio_out_enabled=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(stop_secs=config.VAD_STOP_SECS)
            ),
        ),
    )

    # ── STT ───────────────────────────────────────────────
    if config.STT_MODE == "local":
        try:
            from pipecat.services.whisper.stt import WhisperSTTService

            stt = WhisperSTTService(
                model=config.WHISPER_MODEL,
                device=config.WHISPER_DEVICE,
                compute_type=config.WHISPER_COMPUTE_TYPE,
                no_speech_prob=0.4,
            )
            logger.info(
                f"STT: local whisper ({config.WHISPER_MODEL}, {config.WHISPER_DEVICE})"
            )
        except ImportError:
            logger.warning(
                "WhisperSTTService not available, falling back to HTTP mode"
            )
            from whisper_http_stt import WhisperHTTPSTTService

            stt = WhisperHTTPSTTService(
                base_url=config.STT_BASE_URL,
                api_key=config.STT_API_KEY,
                language=config.WHISPER_LANGUAGE,
                sample_rate=config.STT_SAMPLE_RATE,
                prompt=config.STT_PROMPT,
                min_audio_secs=config.STT_MIN_AUDIO_SECS,
                min_audio_rms=config.STT_MIN_AUDIO_RMS,
            )
    else:
        from whisper_http_stt import WhisperHTTPSTTService

        stt = WhisperHTTPSTTService(
            base_url=config.STT_BASE_URL,
            api_key=config.STT_API_KEY,
            language=config.WHISPER_LANGUAGE,
            sample_rate=config.STT_SAMPLE_RATE,
            prompt=config.STT_PROMPT,
            min_audio_secs=config.STT_MIN_AUDIO_SECS,
            min_audio_rms=config.STT_MIN_AUDIO_RMS,
        )
        logger.info(f"STT: HTTP mode → {config.STT_BASE_URL}")

    logger.info("Voice input layer mode enabled: STT only (LLM/TTS handled by AssistantPage)")

    # ── Pipeline ──────────────────────────────────────────
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            transport.output(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=config.ALLOW_INTERRUPTIONS,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # ── Event Handlers ────────────────────────────────────
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected: {client}")
        # 等待使用者第一句語音再觸發 LLM，避免送出空的 user message。

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected: {client}")
        await task.cancel()

    # ── Run ───────────────────────────────────────────────
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)

    logger.info("Voice bot pipeline finished")
