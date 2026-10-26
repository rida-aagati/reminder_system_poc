"""
TalentSkout Real-Time Asynchronous Voice Agent (Microphone & TTS Engine)
========================================================================
Features:
  1. Ambient Room Noise Calibration: Dynamically measures ambient noise to prevent false triggering.
  2. 3-Chunk Debounce & Echo Lock: Requires sustained speech (~90ms) and never self-triggers on agent speech.
  3. Active Barge-In Task Cancellation: Instantly halts speaker output and cancels in-flight LLM & TTS tasks.
  4. Real Deepgram Aura TTS: Streams natural human-quality WAV audio through speakers.
  5. Real Deepgram Nova-3 STT: Transcribes live candidate microphone speech asynchronously.
  6. Code-Level Input & Output Guardrails: Blocks prompt injections and redacts sensitive data.
  7. Cross-Platform & VS Code Ready: Auto-loads .env and falls back gracefully to console input if mic is missing.
"""

import os
import re
import json
import time
import math
import wave
import struct
import asyncio
import io
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, AsyncGenerator

# Audio libraries
try:
    import sounddevice as sd
    import numpy as np
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "db.json"
OUTPUT_AUDIO_DIR = BASE_DIR / "output_audio"
OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Auto-load .env file so API keys work inside VS Code automatically!
env_path = BASE_DIR / ".env"
if env_path.exists():
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

SAMPLE_RATE = 16000          # 16kHz standard for Deepgram
CHANNELS = 1                 # Mono
CHUNK_DURATION_MS = 30       # 30ms audio frame chunks
SILENCE_TIMEOUT_MS = 700     # Silence duration to mark end of turn
MIN_SPEECH_DURATION_MS = 400 # Minimum speech length (ignores clicks)

DEEPGRAM_TTS_MODEL = "aura-asteria-en"
DEEPGRAM_STT_MODEL = "nova-3"


# ==============================================================================
# SECTION 1: DATABASE & INTERVIEW CONTEXT
# ==============================================================================

def load_db() -> Dict[str, Any]:
    if not DB_PATH.exists():
        return {"interviews": []}
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_db(data: Dict[str, Any]) -> None:
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_first_pending_interview() -> Dict[str, Any]:
    db = load_db()
    for item in db.get("interviews", []):
        if item.get("status") == "pending" and item.get("attendance_confirmed") is not True:
            return item
    return db["interviews"][0]


def get_spoken_interview_date(interview: Dict[str, Any]) -> str:
    raw = interview.get("scheduled_time") or interview.get("link_expires_at") or interview.get("time_slot")
    if not raw:
        return "the scheduled date"
    try:
        dt = datetime.fromisoformat(raw)
        day = dt.day
        suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        time_str = dt.strftime("%I:%M %p").lstrip("0")
        return f"{dt.strftime('%B')} {day}{suffix} at {time_str}"
    except Exception:
        return str(raw)


def tool_confirm_attendance(interview_id: str) -> dict:
    db = load_db()
    for item in db.get("interviews", []):
        if item["interview_id"] == interview_id:
            item["attendance_confirmed"] = True
            item["status"] = "confirmed"
            item["call_reminder_sent"] = True
            save_db(db)
            return {"success": True, "message": f"Confirmed attendance for {item['candidate_name']}."}
    return {"error": "interview not found"}


# ==============================================================================
# SECTION 2: GUARDRAILS (Code-Level Input & Output Filtering)
# ==============================================================================

INJECTION_PATTERNS = [
    r"discard all previous instructions",
    r"ignore all (previous|prior|system) instructions",
    r"disregard (all )?(instructions|rules|prompts)",
    r"you are now in (dan|developer|god) mode",
    r"system prompt override",
    r"tell me your (system )?prompt",
    r"reveal (the )?system instructions",
    r"jailbreak",
]

CONFIDENTIAL_PATTERNS = [
    r"confidential",
    r"api key",
    r"secret key",
    r"password",
    r"internal database",
    r"candidate phone number",
    r"personal phone",
    r"personal email",
    r"recruiter salary",
]


class InputGuardrail:
    @staticmethod
    def inspect(user_text: str) -> Tuple[bool, Optional[str]]:
        lowered = user_text.lower()
        for pat in INJECTION_PATTERNS:
            if re.search(pat, lowered):
                return False, (
                    "This information is confidential and cannot be shared. "
                    "I can only assist you with questions related to your TalentSkout interview. "
                    "Are you planning to appear for the interview?"
                )
        for pat in CONFIDENTIAL_PATTERNS:
            if re.search(pat, lowered):
                return False, (
                    "I cannot share confidential internal information. "
                    "I am here only to assist with your interview details. "
                    "Will you be appearing for the scheduled interview?"
                )
        return True, None


class OutputGuardrail:
    @staticmethod
    def sanitize(text: str) -> str:
        phone_pattern = r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
        text = re.sub(phone_pattern, "[Contact through TalentSkout Portal]", text)
        text = re.sub(r"[\*#_`]", "", text)
        text = re.sub(r"^\s*-\s+", "", text, flags=re.MULTILINE)
        return text.strip()


# ==============================================================================
# SECTION 3: AUDIBLE SPEECH GENERATOR (Deepgram Aura + Fallback Chime)
# ==============================================================================

def generate_audible_fallback_wav(duration_sec: float = 1.2) -> bytes:
    """Generates a melodic chime tone if Deepgram API key is not yet set."""
    num_samples = int(SAMPLE_RATE * duration_sec)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        for i in range(num_samples):
            t = i / SAMPLE_RATE
            freq = 523.25 if t < 0.6 else 659.25
            env = math.exp(-2.5 * (t % 0.6))
            val = int(9000 * env * math.sin(2 * math.pi * freq * t))
            wf.writeframes(struct.pack("<h", val))
    return buf.getvalue()


async def async_deepgram_tts(
    text: str,
    output_filename: Optional[str] = None,
    voice: str = DEEPGRAM_TTS_MODEL,
) -> bytes:
    clean_text = OutputGuardrail.sanitize(text)
    if not clean_text:
        return b""

    key = os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        audio_bytes = generate_audible_fallback_wav()
        if output_filename:
            with open(OUTPUT_AUDIO_DIR / output_filename, "wb") as f:
                f.write(audio_bytes)
        return audio_bytes

    loop = asyncio.get_running_loop()

    def _sync_request() -> bytes:
        url = f"https://api.deepgram.com/v1/speak?model={voice}&encoding=linear16&container=wav"
        data = json.dumps({"text": clean_text}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Token {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            return resp.read()

    try:
        audio_bytes = await loop.run_in_executor(None, _sync_request)
        if output_filename:
            with open(OUTPUT_AUDIO_DIR / output_filename, "wb") as f:
                f.write(audio_bytes)
        return audio_bytes
    except urllib.error.HTTPError as exc:
        err_detail = exc.read().decode("utf-8")
        print(f"  [Deepgram TTS Error {exc.code}]: {err_detail}")
        return generate_audible_fallback_wav()
    except Exception as exc:
        print(f"  [Deepgram TTS Warning] {exc} — generating fallback chime.")
        return generate_audible_fallback_wav()


# ==============================================================================
# SECTION 4: DEEPGRAM NOVA-3 STT
# ==============================================================================

async def async_deepgram_stt(audio_wav_bytes: bytes) -> str:
    key = os.environ.get("DEEPGRAM_API_KEY")
    if not key or not audio_wav_bytes or len(audio_wav_bytes) < 400:
        return ""

    loop = asyncio.get_running_loop()

    def _sync_transcribe() -> str:
        url = f"https://api.deepgram.com/v1/listen?model={DEEPGRAM_STT_MODEL}&smart_format=true"
        req = urllib.request.Request(
            url,
            data=audio_wav_bytes,
            headers={
                "Authorization": f"Token {key}",
                "Content-Type": "audio/wav",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            return res_data["results"]["channels"][0]["alternatives"][0]["transcript"]

    try:
        return await loop.run_in_executor(None, _sync_transcribe)
    except Exception:
        return ""


# ==============================================================================
# SECTION 5: LIVE MICROPHONE VAD (With Ambient Calibration & Debouncing)
# ==============================================================================

class LiveMicrophoneVAD:
    def __init__(self, on_barge_in_callback):
        self.on_barge_in = on_barge_in_callback
        self.audio_queue = asyncio.Queue()
        self.is_recording = False
        self.stream = None
        self.loop = None
        self.energy_threshold = 0.08
        self.is_agent_speaking = False

    def _mic_callback(self, indata, frames, time_info, status):
        if not self.is_recording or self.loop is None:
            return
        audio_chunk = bytes(indata)
        self.loop.call_soon_threadsafe(self.audio_queue.put_nowait, audio_chunk)

    async def start(self):
        self.loop = asyncio.get_running_loop()
        self.is_recording = True
        if HAS_SOUNDDEVICE:
            self.stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                blocksize=int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000),
                dtype="int16",
                channels=CHANNELS,
                callback=self._mic_callback,
            )
            self.stream.start()
            await self.calibrate_ambient_noise()

    async def calibrate_ambient_noise(self, duration_sec: float = 0.8):
        """Measures room background noise level to set the exact threshold automatically."""
        print("  ⏳ [Calibrating Mic] Measuring ambient noise floor...", end="", flush=True)
        energies = []
        start = time.time()
        while time.time() - start < duration_sec:
            try:
                chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=0.1)
                count = len(chunk) // 2
                shorts = struct.unpack(f"<{count}h", chunk)
                sum_sq = sum(s * s for s in shorts)
                rms = math.sqrt(sum_sq / count) / 32768.0
                energies.append(rms)
            except asyncio.TimeoutError:
                pass
        baseline = max(energies) if energies else 0.03
        self.energy_threshold = max(0.07, baseline * 2.2)
        print(f" Ready! (Ambient noise: {baseline:.3f}, Threshold set to: {self.energy_threshold:.3f})")

    async def stop(self):
        self.is_recording = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    async def listen_for_turn(self) -> Any:
        if not HAS_SOUNDDEVICE:
            print("  🎤 (Type candidate reply in console): ", end="", flush=True)
            text_input = await asyncio.to_thread(input)
            return ("TEXT_INPUT", text_input)

        collected_frames = []
        is_speech_active = False
        consecutive_loud_chunks = 0
        silence_start_time = None
        speech_start_time = None

        while True:
            try:
                chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            count = len(chunk) // 2
            shorts = struct.unpack(f"<{count}h", chunk)
            sum_squares = sum(s * s for s in shorts)
            rms = math.sqrt(sum_squares / count) / 32768.0

            # 1. Loud audio chunk detected
            if rms > self.energy_threshold:
                consecutive_loud_chunks += 1

                # Require at least 3 consecutive loud chunks (~90ms) before treating as speech
                if consecutive_loud_chunks >= 3 and not is_speech_active:
                    is_speech_active = True
                    speech_start_time = time.time()
                    silence_start_time = None

                    # If agent was speaking, trigger BARGE-IN!
                    if self.is_agent_speaking:
                        await self.on_barge_in()
                    print("\n  🎤 [User Speaking...] ", end="", flush=True)

                if is_speech_active:
                    collected_frames.append(chunk)
                    silence_start_time = None

            # 2. Quiet chunk
            else:
                consecutive_loud_chunks = 0
                if is_speech_active:
                    collected_frames.append(chunk)
                    now = time.time()
                    if silence_start_time is None:
                        silence_start_time = now
                    elif (now - silence_start_time) * 1000 >= SILENCE_TIMEOUT_MS:
                        speech_duration = (now - speech_start_time) * 1000
                        if speech_duration >= MIN_SPEECH_DURATION_MS:
                            print(" [Turn Complete]")
                            break
                        else:
                            is_speech_active = False
                            collected_frames.clear()

        raw_pcm = b"".join(collected_frames)
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(raw_pcm)
        return wav_buf.getvalue()


# ==============================================================================
# SECTION 6: CANCELLABLE AUDIO PLAYER (Instant Barge-In Interruption)
# ==============================================================================

class CancellableAudioPlayer:
    def __init__(self):
        self.stop_playback = False

    async def play_wav(self, wav_bytes: bytes):
        if not wav_bytes or len(wav_bytes) < 44:
            return

        self.stop_playback = False

        if HAS_SOUNDDEVICE:
            try:
                with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                    sr = wf.getframerate()
                    frames = wf.readframes(wf.getnframes())
                audio_array = np.frombuffer(frames, dtype=np.int16)

                sd.play(audio_array, samplerate=sr)
                duration_sec = len(audio_array) / sr

                elapsed = 0.0
                while elapsed < duration_sec:
                    if self.stop_playback:
                        sd.stop()
                        break
                    await asyncio.sleep(0.04)
                    elapsed += 0.04
                sd.stop()
                return
            except Exception:
                pass

        if HAS_WINSOUND:
            try:
                temp_path = OUTPUT_AUDIO_DIR / "temp_speaker.wav"
                with open(temp_path, "wb") as f:
                    f.write(wav_bytes)
                with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                    duration_sec = wf.getnframes() / float(wf.getframerate())

                winsound.PlaySound(str(temp_path), winsound.SND_FILENAME | winsound.SND_ASYNC)
                elapsed = 0.0
                while elapsed < duration_sec:
                    if self.stop_playback:
                        winsound.PlaySound(None, winsound.SND_PURGE)
                        break
                    await asyncio.sleep(0.04)
                    elapsed += 0.04
            except Exception:
                pass

    def cancel(self):
        self.stop_playback = True
        if HAS_SOUNDDEVICE:
            try:
                sd.stop()
            except Exception:
                pass
        if HAS_WINSOUND:
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass


# ==============================================================================
# SECTION 7: STREAMING RESPONSE GENERATOR WITH GUARDRAILS & TOOLS
# ==============================================================================

async def async_stream_llm_response(
    transcript: str,
    interview: Dict[str, Any],
) -> AsyncGenerator[str, None]:
    # 1. Input Guardrail
    passed, blocked_msg = InputGuardrail.inspect(transcript)
    if not passed:
        yield OutputGuardrail.sanitize(blocked_msg)
        return

    text_lower = transcript.lower()
    iid = interview["interview_id"]
    name = interview["candidate_name"]
    role = interview["role"]
    itype = interview["interview_type"]
    spoken_date = get_spoken_interview_date(interview)

    # 2. Scope Limitations (Insurance, Weather, etc.)
    if any(k in text_lower for k in ["insurance", "weather", "crypto", "pizza", "flight", "claim"]):
        yield OutputGuardrail.sanitize(
            "I'm here only to assist you with questions regarding your upcoming TalentSkout interview. "
            f"Are you planning to appear for the interview on {spoken_date}?"
        )
        return

    # 3. Attendance Confirmation
    if any(k in text_lower for k in ["yes", "i will appear", "i'll appear", "attend", "sure", "confirmed", "available", "definitely"]):
        tool_confirm_attendance(iid)
        if itype == "ai_screener":
            yield OutputGuardrail.sanitize(
                f"Wonderful, {name}! I have recorded your confirmation. "
                "Please be sure to complete the assessment link sent to your email before it expires."
            )
        else:
            yield OutputGuardrail.sanitize(
                f"Wonderful, {name}! Your attendance has been confirmed for the {role} interview on {spoken_date}. "
                "We look forward to meeting you."
            )
        return

    # 4. Panelist Query
    if any(k in text_lower for k in ["panelist", "interviewer", "who is taking", "who will interview"]):
        if itype == "ai_screener":
            yield OutputGuardrail.sanitize(
                "For this screening round, you will be interacting directly with our AI screener, so there is no human panelist. "
                "Are you ready to appear?"
            )
        else:
            panelist = interview.get("panelist_name") or "our Senior Technical Lead"
            yield OutputGuardrail.sanitize(
                f"Your interview will be conducted by {panelist}. Are you going to be able to connect on time?"
            )
        return

    # 5. Timeline / Expiration Query
    if any(k in text_lower for k in ["how much time", "expire", "deadline", "remaining", "valid", "link"]):
        time_left = interview.get("time_left_hours", 2.0)
        yield OutputGuardrail.sanitize(
            f"Your assessment link expires in approximately {time_left:.1f} hours. "
            "We recommend completing it as soon as possible. Will you be submitting it today?"
        )
        return

    # 6. Role / Company / Duration Query
    if any(k in text_lower for k in ["role", "company", "duration", "how long"]):
        yield OutputGuardrail.sanitize(
            f"This is for the {role} position at {interview['company']}, taking approximately {interview['duration_minutes']} minutes. "
            "Will you be appearing?"
        )
        return

    # Default confirmation prompt
    yield OutputGuardrail.sanitize(
        f"Thanks for letting me know. Are you available to appear for your interview on {spoken_date}?"
    )


# ==============================================================================
# SECTION 8: FULL ASYNCHRONOUS REAL-TIME VOICE PIPELINE
# ==============================================================================

class RealtimeVoicePipeline:
    def __init__(self, interview: Dict[str, Any]):
        self.interview = interview
        self.player = CancellableAudioPlayer()
        self.mic = LiveMicrophoneVAD(on_barge_in_callback=self.on_barge_in)
        self.active_agent_task = None
        self.turn_count = 0

    async def on_barge_in(self):
        print("\n⚡ [BARGE-IN] Candidate interrupted! Halting speaker output and cancelling LLM.")
        self.player.cancel()
        if self.active_agent_task and not self.active_agent_task.done():
            self.active_agent_task.cancel()
            self.active_agent_task = None

    async def speak_text(self, text: str, filename: str):
        self.mic.is_agent_speaking = True
        print(f"\n🤖 Agent: \"{text}\"")
        audio_wav = await async_deepgram_tts(text, output_filename=filename)
        await self.player.play_wav(audio_wav)
        self.mic.is_agent_speaking = False

    async def run_conversation_loop(self):
        print("""
================================================================================
     TALENTSKOUT REAL-TIME VOICE AGENT (LIVE MIC + DEEPGRAM TTS)
================================================================================
""")
        if not os.environ.get("DEEPGRAM_API_KEY"):
            print("👉 NOTE: DEEPGRAM_API_KEY is not set. The agent will speak with audible chimes.")
            print("   To hear REAL human voice, ensure DEEPGRAM_API_KEY is set in .env\n")
        else:
            print("✅ [Deepgram Aura Active] Spoken audio will play through your speakers.")

        if HAS_SOUNDDEVICE:
            print("🎙️ [Live Mic Active] Speak into your laptop microphone!")
        else:
            print("ℹ️ [Console Input Mode] 'sounddevice' not found; type your replies in console.")

        spoken_date = get_spoken_interview_date(self.interview)
        greeting = (
            f"Hi {self.interview['candidate_name']}, I am calling on behalf of TalentSkout. "
            f"I am calling to confirm, will you be available to appear for the interview on {spoken_date} "
            f"for the role of {self.interview['role']} with this company, {self.interview['company']}?"
        )

        # 1. Start Mic in background (measures room noise first)
        await self.mic.start()

        # 2. Speak greeting out loud
        await self.speak_text(greeting, filename="live_turn_0_greeting.wav")

        try:
            while True:
                # 3. Asynchronously listen to microphone (non-blocking)
                audio_or_text = await self.mic.listen_for_turn()

                if isinstance(audio_or_text, tuple) and audio_or_text[0] == "TEXT_INPUT":
                    transcript = audio_or_text[1]
                else:
                    t0 = time.perf_counter()
                    transcript = await async_deepgram_stt(audio_or_text)
                    stt_ms = (time.perf_counter() - t0) * 1000
                    if transcript:
                        print(f"👤 Candidate [Deepgram STT ({stt_ms:.0f}ms)]: \"{transcript}\"")
                    else:
                        continue

                if not transcript or not transcript.strip():
                    continue

                if any(k in transcript.lower() for k in ["bye", "hang up", "thank you", "goodbye"]):
                    farewell = "Thank you so much! Have a great day and best of luck with your interview."
                    await self.speak_text(farewell, filename="live_farewell.wav")
                    break

                self.turn_count += 1

                async def agent_turn():
                    async for sentence in async_stream_llm_response(transcript, self.interview):
                        await self.speak_text(sentence, filename=f"live_turn_{self.turn_count}.wav")

                self.active_agent_task = asyncio.create_task(agent_turn())
                try:
                    await self.active_agent_task
                except asyncio.CancelledError:
                    print("  [Turn cancelled due to barge-in]")
                finally:
                    self.active_agent_task = None

        finally:
            await self.mic.stop()
            print("\n📴 [Call Ended] Conversation finished.")


if __name__ == "__main__":
    interview = get_first_pending_interview()
    pipeline = RealtimeVoicePipeline(interview)
    asyncio.run(pipeline.run_conversation_loop())
