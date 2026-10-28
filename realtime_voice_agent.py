"""
TalentSkout Real-Time Asynchronous Voice Outreach Agent
======================================================
Production-Grade Full-Duplex Voice Agent Architecture:
  1. Continuous Duplex Audio: The microphone listener runs continuously in parallel
     with audio playback from the very first second (even during the greeting!).
  2. Active Instant Barge-In: When the candidate speaks while the agent is talking
     (intro or reply), the speaker is silenced within 50ms and the in-flight response
     is cancelled. The candidate's utterance is seamlessly captured as their turn.
  3. Context-Aware Dialog Engine:
     - Handles "I have a few questions" / "I have doubts" with warm, helpful guidance.
     - Answers role details, interview format, duration, deadlines, prep tips, and panelists.
     - Executes confirm_attendance tool and updates db.json.
     - Strictly enforces code-level input & output guardrails (prompt injection & data privacy).
  4. Real Deepgram Aura TTS (aura-asteria-en) + Nova-3 STT.
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

# Audio playback and recording libraries
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
SILENCE_TIMEOUT_MS = 650     # Silence duration to mark end of turn
MIN_SPEECH_DURATION_MS = 350 # Minimum speech length (ignores tiny clicks/coughs)

DEEPGRAM_TTS_MODEL = "aura-asteria-en"
DEEPGRAM_STT_MODEL = "nova-3"

# ==============================================================================
# VAD & BARGE-IN SENSITIVITY TUNING (Easily change sensitivity!)
# ==============================================================================
# SENSITIVITY PRESETS:
#   "low"    -> Highly resilient to background noise/typing/fans (Requires firm voice to interrupt)
#   "medium" -> Balanced default (Filters normal room noise, interrupts on clear voice)
#   "high"   -> Sensitive (Interrupts on soft voices, recommended with headphones)
SENSITIVITY_MODE = os.environ.get("VAD_SENSITIVITY", "medium").lower()

PRESETS = {
    "low": {
        "barge_in_threshold": 0.28,      # Minimum volume to interrupt agent
        "barge_in_frames": 6,             # Sustained duration (~180ms)
        "quiet_threshold": 0.085,         # Volume when waiting for candidate
        "quiet_frames": 3,
    },
    "medium": {
        "barge_in_threshold": 0.22,      # Minimum volume to interrupt agent
        "barge_in_frames": 5,             # Sustained duration (~150ms)
        "quiet_threshold": 0.075,         # Volume when waiting for candidate
        "quiet_frames": 3,
    },
    "high": {
        "barge_in_threshold": 0.14,
        "barge_in_frames": 3,
        "quiet_threshold": 0.05,
        "quiet_frames": 2,
    },
}

CFG = PRESETS.get(SENSITIVITY_MODE, PRESETS["medium"])
BARGE_IN_THRESHOLD = CFG["barge_in_threshold"]
BARGE_IN_CONSECUTIVE_FRAMES = CFG["barge_in_frames"]
QUIET_SPEECH_THRESHOLD = CFG["quiet_threshold"]
QUIET_CONSECUTIVE_FRAMES = CFG["quiet_frames"]

# Optional Anthropic Claude LLM
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")
async_anthropic_client = None
if ANTHROPIC_KEY:
    try:
        from anthropic import AsyncAnthropic
        async_anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_KEY)
    except Exception:
        async_anthropic_client = None


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


def tool_record_withdrawal(interview_id: str, reason: str = "Candidate withdrawn") -> dict:
    db = load_db()
    for item in db.get("interviews", []):
        if item["interview_id"] == interview_id:
            item["attendance_confirmed"] = False
            item["status"] = "withdrawn"
            item["notes"] = reason
            save_db(db)
            return {"success": True, "message": f"Recorded withdrawal for {item['candidate_name']}."}
    return {"error": "interview not found"}


# ==============================================================================
# SECTION 2: CODE-LEVEL GUARDRAILS (Prompt Injection & PII Redaction)
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
    """Generates a pleasant melodic chime if API key is not yet set."""
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
        for attempt in range(2):
            try:
                with urllib.request.urlopen(req, timeout=12) as resp:
                    return resp.read()
            except urllib.error.HTTPError as he:
                if he.code in (429, 503) and attempt == 0:
                    time.sleep(0.5)
                    continue
                raise

    try:
        audio_bytes = await loop.run_in_executor(None, _sync_request)
        if output_filename:
            with open(OUTPUT_AUDIO_DIR / output_filename, "wb") as f:
                f.write(audio_bytes)
        return audio_bytes
    except Exception as exc:
        print(f"  [Deepgram TTS Warning] {exc} — generating fallback chime.")
        return generate_audible_fallback_wav()


# ==============================================================================
# SECTION 4: DEEPGRAM NOVA-3 STT (Speech-to-Text)
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
# SECTION 5: CANCELLABLE AUDIO PLAYER (Instant Barge-In Interruption)
# ==============================================================================

class CancellableAudioPlayer:
    """Non-blocking audio player that can be aborted in <50ms upon user barge-in."""

    def __init__(self):
        self.is_playing = False
        self.stop_playback = False

    async def play_wav(self, wav_bytes: bytes):
        if not wav_bytes or len(wav_bytes) < 44:
            return

        self.stop_playback = False
        self.is_playing = True

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
                    await asyncio.sleep(0.03)
                    elapsed += 0.03
                sd.stop()
            except Exception:
                pass
            finally:
                self.is_playing = False

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
                    await asyncio.sleep(0.03)
                    elapsed += 0.03
            except Exception:
                pass
            finally:
                self.is_playing = False

    def cancel(self):
        self.stop_playback = True
        self.is_playing = False
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
# SECTION 6: CONTINUOUS DUPLEX MICROPHONE LISTENER WITH INSTANT BARGE-IN
# ==============================================================================

class ContinuousMicrophoneVAD:
    """
    Runs a continuous background listener.
    Detects user speech at all times (including during the introduction)
    and instantly cancels speaker output when the user begins speaking.
    """

    def __init__(self, pipeline: "RealtimeVoicePipeline"):
        self.pipeline = pipeline
        self.raw_queue = asyncio.Queue()
        self.is_recording = False
        self.stream = None
        self.loop = None
        self.baseline_noise = 0.02
        self.quiet_threshold = QUIET_SPEECH_THRESHOLD
        self.barge_in_threshold = BARGE_IN_THRESHOLD
        self.rms_history = [0.0, 0.0, 0.0]

    def _mic_callback(self, indata, frames, time_info, status):
        if not self.is_recording or self.loop is None:
            return
        chunk = bytes(indata)
        self.loop.call_soon_threadsafe(self.raw_queue.put_nowait, chunk)

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
            await self.calibrate_ambient()

    async def calibrate_ambient(self, duration_sec: float = 1.0):
        print("  ⏳ [Calibrating Mic] Measuring ambient noise floor...", end="", flush=True)
        energies = []
        start = time.time()
        while time.time() - start < duration_sec:
            try:
                chunk = await asyncio.wait_for(self.raw_queue.get(), timeout=0.1)
                count = len(chunk) // 2
                shorts = struct.unpack(f"<{count}h", chunk)
                sum_sq = sum(s * s for s in shorts)
                rms = math.sqrt(sum_sq / count) / 32768.0
                energies.append(rms)
            except asyncio.TimeoutError:
                pass
        self.baseline_noise = max(energies) if energies else 0.02
        # Dynamic thresholding: Quiet threshold is for candidate answers; Barge-in threshold is for interrupting agent
        self.quiet_threshold = max(QUIET_SPEECH_THRESHOLD, self.baseline_noise * 2.2)
        self.barge_in_threshold = max(BARGE_IN_THRESHOLD, self.baseline_noise * 4.8)
        print(f" Ready!\n     Ambient Floor: {self.baseline_noise:.3f} | Mode: {SENSITIVITY_MODE.upper()} | Quiet Cutoff: {self.quiet_threshold:.3f} | Barge-In Trigger: {self.barge_in_threshold:.3f}")

    async def stop(self):
        self.is_recording = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    async def listen_loop(self):
        """Continuously monitors microphone and pushes completed candidate turns to user_turn_queue."""
        if not HAS_SOUNDDEVICE:
            while self.is_recording:
                print("\n  🎤 (Type your reply): ", end="", flush=True)
                text = await asyncio.to_thread(input)
                if text and text.strip():
                    if self.pipeline.player.is_playing:
                        await self.pipeline.handle_barge_in()
                    await self.pipeline.user_turn_queue.put(("TEXT", text.strip()))
            return

        collected_chunks = []
        is_speech_active = False
        consecutive_loud = 0
        silence_start = None
        speech_start = None

        while self.is_recording:
            try:
                chunk = await asyncio.wait_for(self.raw_queue.get(), timeout=0.08)
            except asyncio.TimeoutError:
                continue

            count = len(chunk) // 2
            shorts = struct.unpack(f"<{count}h", chunk)
            sum_sq = sum(s * s for s in shorts)
            raw_rms = math.sqrt(sum_sq / count) / 32768.0

            # 3-frame rolling smoothed energy to filter out random acoustic clicks/pops
            self.rms_history.pop(0)
            self.rms_history.append(raw_rms)
            rms = 0.5 * self.rms_history[2] + 0.3 * self.rms_history[1] + 0.2 * self.rms_history[0]

            agent_is_speaking = self.pipeline.player.is_playing
            active_threshold = self.barge_in_threshold if agent_is_speaking else self.quiet_threshold
            required_frames = BARGE_IN_CONSECUTIVE_FRAMES if agent_is_speaking else QUIET_CONSECUTIVE_FRAMES

            if rms > active_threshold:
                consecutive_loud += 1

                # Require sustained vocal energy (e.g. 5 frames ~150ms during agent speech)
                if consecutive_loud >= required_frames:
                    if not is_speech_active:
                        is_speech_active = True
                        speech_start = time.time()
                        silence_start = None

                        # ⚡ INSTANT BARGE-IN INTERRUPT!
                        if agent_is_speaking:
                            await self.pipeline.handle_barge_in()

                        print("\n  🎤 [User Speaking...] ", end="", flush=True)

                    collected_chunks.append(chunk)
                    silence_start = None

            else:
                consecutive_loud = 0
                if is_speech_active:
                    collected_chunks.append(chunk)
                    now = time.time()
                    if silence_start is None:
                        silence_start = now
                    elif (now - silence_start) * 1000 >= SILENCE_TIMEOUT_MS:
                        # Turn complete!
                        duration_ms = (now - speech_start) * 1000
                        if duration_ms >= MIN_SPEECH_DURATION_MS:
                            print(" [Turn Complete]")
                            raw_pcm = b"".join(collected_chunks)
                            wav_buf = io.BytesIO()
                            with wave.open(wav_buf, "wb") as wf:
                                wf.setnchannels(CHANNELS)
                                wf.setsampwidth(2)
                                wf.setframerate(SAMPLE_RATE)
                                wf.writeframes(raw_pcm)

                            # Push to pipeline turn queue
                            await self.pipeline.user_turn_queue.put(("AUDIO", wav_buf.getvalue()))
                        else:
                            # Ignored brief click/cough
                            pass

                        is_speech_active = False
                        collected_chunks.clear()
                        silence_start = None
                        speech_start = None


# ==============================================================================
# SECTION 7: REAL ASYNCHRONOUS STREAMING LLM & OUTPUT GUARDRAIL
# ==============================================================================

def build_agent_system_prompt(interview: Dict[str, Any]) -> str:
    itype = interview.get("interview_type", "ai_screener")
    spoken_date = get_spoken_interview_date(interview)
    role = interview.get("role", "Candidate")
    company = interview.get("company", "TalentSkout")
    name = interview.get("candidate_name", "Candidate")
    duration = interview.get("duration_minutes", 30)
    time_left = interview.get("time_left_hours", 2.0)
    panelist = interview.get("panelist_name")
    recruiter = interview.get("recruiter_name", "your TalentSkout recruiter")
    email = interview.get("email", "your registered email")

    panelist_desc = (
        f"Your interview will be conducted by {panelist}."
        if itype == "panel_round"
        else "This is an automated 30-minute online screening assessment, so there is no human panelist for this stage."
    )

    return f"""You are the official AI Voice Outreach Coordinator calling on behalf of TalentSkout to confirm candidate attendance for an upcoming interview.

CANDIDATE & INTERVIEW DETAILS:
- Candidate Name: {name}
- Role Applied: {role}
- Hiring Company: {company}
- Interview Stage: {"AI Screening Assessment (link-based)" if itype == "ai_screener" else "Technical Panel Interview"}
- Scheduled Time: {spoken_date}
- Duration: approximately {duration} minutes
- Assessment Link Validity: approximately {time_left:.1f} hours remaining before link expires
- Panelist: {panelist_desc}
- Recruiter Contact: {recruiter}
- Candidate Email: {email}

YOUR CORE OBJECTIVES:
1. Confirm if the candidate will be available to appear for their interview on {spoken_date}.
2. If the candidate confirms (e.g., "Yes", "I will appear", "Sure", "I will attend", "Confirmed"), warmly record their confirmation and tell them to check their email for the meeting link.
3. If the candidate has questions (e.g., "I have a few questions", "What is the format?", "Who is interviewing me?", "How should I prepare?"), answer naturally, warmly, and concisely using ONLY the facts above.
4. If the candidate asks to reschedule or withdraw, be understanding and inform them that their recruiter {recruiter} will follow up with options.
5. If the candidate asks unrelated/off-topic questions (e.g., car insurance, weather, flight booking, crypto, food), politely decline:
   "I am here only to assist with questions regarding your upcoming TalentSkout interview. Are you planning to appear for your interview?"

STRICT SPOKEN VOICE GUIDELINES:
- Output only natural spoken English sentences suitable for text-to-speech.
- Keep replies brief and conversational (1 to 2 short sentences per response).
- NEVER use markdown, asterisks, bullet points, hashtags, emojis, or quotes.
- Never invent information not present in the details above.
- Never disclose private personal phone numbers or confidential internal data.
"""


def resolve_llm_provider() -> Tuple[str, str, str, str]:
    """
    Detects which LLM provider key is present in environment / .env.
    Returns: (provider_name, api_key, model_name, api_url)
    """
    # 1. Groq (Ultra-fast 100ms TTFB streaming - Ideal for real-time voice!)
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        return ("groq", groq_key, "llama-3.3-70b-versatile", "https://api.groq.com/openai/v1/chat/completions")

    # 2. OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        return ("openai", openai_key, "gpt-4o-mini", "https://api.openai.com/v1/chat/completions")

    # 3. Anthropic Claude
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        return ("anthropic", anthropic_key, "claude-haiku-5-5", "https://api.anthropic.com/v1/messages")

    # 4. Google Gemini
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        return ("gemini", gemini_key, "gemini-2.0-flash", f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent?alt=sse&key={gemini_key}")

    return ("none", "", "", "")


def sync_stream_openai_compatible(url: str, api_key: str, model: str, system: str, messages: List[Dict[str, str]]):
    """Zero-dependency HTTP SSE streaming for OpenAI and Groq APIs."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "stream": True,
        "max_tokens": 120,
        "temperature": 0.3,
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=12) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="ignore").strip()
            if line.startswith("data: "):
                chunk_str = line[6:].strip()
                if chunk_str == "[DONE]":
                    break
                try:
                    data = json.loads(chunk_str)
                    token = data["choices"][0]["delta"].get("content")
                    if token:
                        yield token
                except Exception:
                    continue


def merge_same_role(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Anthropic/Gemini require strictly alternating roles; the pipeline stores one message per sentence."""
    merged: List[Dict[str, Any]] = []
    for m in msgs:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] = (merged[-1]["content"] + " " + m["content"]).strip()
        else:
            merged.append({"role": m["role"], "content": m["content"]})
    return merged


def sync_stream_anthropic(api_key: str, model: str, system: str, messages: List[Dict[str, str]]):
    """Zero-dependency HTTP SSE streaming for Anthropic Claude Messages API."""
    claude_msgs = []
    for m in messages:
        role = "assistant" if m.get("role") in ("assistant", "agent") else "user"
        claude_msgs.append({"role": role, "content": m.get("content") or m.get("text", "")})
    claude_msgs = merge_same_role(claude_msgs)
    payload = json.dumps({
        "model": model,
        "system": system,
        "thinking": {"type": "disabled"},  # voice: hidden thinking eats max_tokens -> silent reply
        "messages": claude_msgs,
        "stream": True,
        "max_tokens": 120,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=12) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="ignore").strip()
            if line.startswith("data: "):
                chunk_str = line[6:].strip()
                try:
                    data = json.loads(chunk_str)
                    if data.get("type") == "content_block_delta":
                        token = data.get("delta", {}).get("text", "")
                        if token:
                            yield token
                except Exception:
                    continue


def sync_stream_gemini(url: str, system: str, messages: List[Dict[str, str]]):
    """Zero-dependency HTTP SSE streaming for Google Gemini API."""
    gemini_contents = []
    for m in messages:
        role = "model" if m.get("role") in ("assistant", "agent") else "user"
        gemini_contents.append({"role": role, "parts": [{"text": m.get("content") or m.get("text", "")}]})
    payload = json.dumps({
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": gemini_contents,
        "generationConfig": {"maxOutputTokens": 120, "temperature": 0.3},
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=12) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="ignore").strip()
            if line.startswith("data: "):
                chunk_str = line[6:].strip()
                try:
                    data = json.loads(chunk_str)
                    for cand in data.get("candidates", []):
                        for part in cand.get("content", {}).get("parts", []):
                            if "text" in part:
                                yield part["text"]
                except Exception:
                    continue


async def async_call_streaming_llm(
    messages: List[Dict[str, str]],
    system_prompt: str,
    interview: Dict[str, Any],
    candidate_utterance: str,
) -> AsyncGenerator[str, None]:
    """
    Asynchronously streams tokens directly from whichever cloud LLM is configured.
    Falls back gracefully to intelligent dynamic streaming if no key is supplied yet.
    """
    provider, api_key, model, endpoint = resolve_llm_provider()

    # If a real cloud LLM provider is configured, stream from it!
    if provider in ("groq", "openai"):
        print(f"  🧠 [LLM Stream] Calling {provider.upper()} ({model}) via async HTTP stream...")
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()
        sentinel = object()

        def worker():
            try:
                for token in sync_stream_openai_compatible(endpoint, api_key, model, system_prompt, messages):
                    loop.call_soon_threadsafe(queue.put_nowait, token)
            except Exception as exc:
                print(f"  [LLM Error {provider}]: {exc}")
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        loop.run_in_executor(None, worker)
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            yield item
        return

    elif provider == "anthropic":
        print(f"  🧠 [LLM Stream] Calling Anthropic ({model}) via async HTTP stream...")
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()
        sentinel = object()

        def worker():
            try:
                for token in sync_stream_anthropic(api_key, model, system_prompt, messages):
                    loop.call_soon_threadsafe(queue.put_nowait, token)
            except Exception as exc:
                print(f"  [LLM Error Anthropic]: {exc}")
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        loop.run_in_executor(None, worker)
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            yield item
        return

    elif provider == "gemini":
        print(f"  🧠 [LLM Stream] Calling Google Gemini ({model}) via async HTTP stream...")
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()
        sentinel = object()

        def worker():
            try:
                for token in sync_stream_gemini(endpoint, system_prompt, messages):
                    loop.call_soon_threadsafe(queue.put_nowait, token)
            except Exception as exc:
                print(f"  [LLM Error Gemini]: {exc}")
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        loop.run_in_executor(None, worker)
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            yield item
        return

    # Fallback when no LLM API key has been added to .env yet
    print("  ⚠️ [LLM Info] No cloud LLM key in .env! (Set GROQ_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY for live models).")
    print("     Streaming via internal dynamic token generator...")
    text_lower = candidate_utterance.lower()
    name = interview.get("candidate_name", "Candidate")
    role = interview.get("role", "Developer")
    spoken_date = get_spoken_interview_date(interview)
    itype = interview.get("interview_type", "ai_screener")
    duration = interview.get("duration_minutes", 30)
    time_left = interview.get("time_left_hours", 2.0)
    panelist = interview.get("panelist_name") or "our Senior Technical Lead"

    if any(k in text_lower for k in ["few questions", "some questions", "have a question", "have questions", "ask something", "doubts"]):
        simulated_text = (
            "Of course! I'd be happy to answer your questions. "
            "You can ask me about the interview format, duration, role details, deadlines, or who your panelist will be. "
            "What would you like to know?"
        )
    elif any(k in text_lower for k in ["yes", "appear", "attend", "sure", "confirmed", "available", "i will"]):
        if itype == "ai_screener":
            simulated_text = f"Wonderful, {name}! I have recorded your confirmation. Please be sure to complete the screening assessment link sent to your email before it expires."
        else:
            simulated_text = f"Wonderful, {name}! Your attendance has been confirmed for the {role} interview on {spoken_date}. We look forward to meeting you."
    elif any(k in text_lower for k in ["format", "what to expect", "how does it work", "structure"]):
        if itype == "ai_screener":
            simulated_text = f"This is an automated 30-minute online screening assessment for {role}. You can take it directly in your browser before the link expires. Will you be appearing today?"
        else:
            simulated_text = f"This is a {duration}-minute live technical interview with {panelist} conducted over video call. Are you ready to appear?"
    elif any(k in text_lower for k in ["panelist", "interviewer", "who is taking"]):
        if itype == "ai_screener":
            simulated_text = "For this initial screening round, you will be interacting directly with our automated AI screener, so there is no human panelist for this stage."
        else:
            simulated_text = f"Your interview will be conducted by {panelist}. Will you be able to connect on time on {spoken_date}?"
    elif any(k in text_lower for k in ["how much time", "expire", "deadline", "remaining"]):
        simulated_text = f"Your assessment link expires in approximately {time_left:.1f} hours. We recommend completing it as soon as possible. Will you be submitting it today?"
    elif any(k in text_lower for k in ["reschedule", "tomorrow", "change time", "not available"]):
        simulated_text = "I completely understand. I will record your reschedule request and notify your recruiter to follow up with alternative time slots."
    else:
        simulated_text = f"I am here to assist you with your upcoming {role} interview on {spoken_date}. Are you planning to appear for the interview?"

    for word in simulated_text.split(" "):
        yield word + " "
        await asyncio.sleep(0.04)


async def async_stream_llm_response(
    transcript: str,
    interview: Dict[str, Any],
    history: List[Dict[str, str]],
) -> AsyncGenerator[str, None]:
    """
    Asynchronous Streaming LLM pipeline:
      1. Pre-LLM Code Input Guardrail (blocks prompt injections & confidential exfiltrations).
      2. Tool check (confirms attendance in database upon candidate confirmation).
      3. Live token streaming from Cloud LLM (Groq / Anthropic / OpenAI / Gemini).
      4. Sentence boundary chunker with real-time Output Guardrail sanitization.
      5. Directly pipes clean sentences to TTS!
    """
    # 1. Code-Level Input Guardrail
    passed, blocked_msg = InputGuardrail.inspect(transcript)
    if not passed:
        yield OutputGuardrail.sanitize(blocked_msg)
        return

    # 2. Scope Limitations Pre-Filter (Insurance, Weather, Loans)
    text_lower = transcript.lower()
    spoken_date = get_spoken_interview_date(interview)
    if any(k in text_lower for k in ["insurance", "weather", "crypto", "pizza", "flight", "car insurance"]):
        yield OutputGuardrail.sanitize(
            "I'm here only to assist you with questions regarding your upcoming TalentSkout interview. "
            f"Are you planning to appear for your interview on {spoken_date}?"
        )
        return

    # 3. Tool Calling: Confirm attendance in database
    iid = interview["interview_id"]
    if any(k in text_lower for k in [
        "yes", "i will appear", "i'll appear", "attend", "sure", "confirmed", 
        "available", "definitely", "i can attend", "count me in", "i will be there"
    ]):
        tool_confirm_attendance(iid)

    # 4. Build System Prompt & Call Asynchronous Streaming LLM
    system_prompt = build_agent_system_prompt(interview)

    sentence_buffer = ""
    async for token in async_call_streaming_llm(history, system_prompt, interview, transcript):
        sentence_buffer += token

        # Sentence boundary detection [.!?] followed by whitespace
        match = re.search(r"([.!?])\s+", sentence_buffer)
        if match:
            end_pos = match.end()
            sentence = sentence_buffer[:end_pos].strip()
            sentence_buffer = sentence_buffer[end_pos:]

            # 5. Output Guardrail: Sanitize each sentence before feeding to TTS!
            sanitized = OutputGuardrail.sanitize(sentence)
            if sanitized:
                yield sanitized

    # Flush any remaining text in buffer
    if sentence_buffer.strip():
        sanitized = OutputGuardrail.sanitize(sentence_buffer.strip())
        if sanitized:
            yield sanitized


# ==============================================================================
# SECTION 8: FULL ASYNCHRONOUS REAL-TIME VOICE PIPELINE
# ==============================================================================

class RealtimeVoicePipeline:
    def __init__(self, interview: Dict[str, Any]):
        self.interview = interview
        self.player = CancellableAudioPlayer()
        self.vad = ContinuousMicrophoneVAD(self)
        self.user_turn_queue = asyncio.Queue()
        self.active_agent_task = None
        self.turn_count = 0
        self.conversation_history = []
        self.is_running = True

    async def handle_barge_in(self):
        """Immediately halts speaker playback and cancels in-flight agent processing."""
        if self.player.is_playing or (self.active_agent_task and not self.active_agent_task.done()):
            print("\n⚡ [BARGE-IN] Candidate interrupted! Silencing speaker and aborting agent turn...")
            self.player.cancel()
            if self.active_agent_task and not self.active_agent_task.done():
                self.active_agent_task.cancel()
                self.active_agent_task = None

    async def speak_text(self, text: str, filename: str):
        """Generates speech via Deepgram Aura and plays through speaker (abortable by barge-in)."""
        print(f"\n🤖 Agent: \"{text}\"")
        audio_wav = await async_deepgram_tts(text, output_filename=filename)
        await self.player.play_wav(audio_wav)

    async def run_conversation_loop(self):
        print("""
================================================================================
     TALENTSKOUT REAL-TIME VOICE AGENT (LIVE MIC + FULL BARGE-IN)
================================================================================
""")
        if not os.environ.get("DEEPGRAM_API_KEY"):
            print("👉 NOTE: DEEPGRAM_API_KEY is not set. The agent will speak with audible chimes.")
            print("   To hear REAL human voice, ensure DEEPGRAM_API_KEY is set in .env\n")
        else:
            print("✅ [Deepgram Aura Active] Spoken audio will play through your speakers.")

        provider, _, model, _ = resolve_llm_provider()
        if provider != "none":
            print(f"🧠 [Active Streaming LLM] Calling {provider.upper()} ({model}) via asynchronous HTTP stream.")
        else:
            print("⚠️ [LLM Info] No cloud LLM key in .env! (Add GROQ_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY for live models).")

        if HAS_SOUNDDEVICE:
            print("🎙️ [Continuous Live Mic Active] Speak anytime — including during the intro to interrupt!")
        else:
            print("ℹ️ [Console Input Mode] 'sounddevice' not found; type your replies in console.")

        spoken_date = get_spoken_interview_date(self.interview)
        greeting = (
            f"Hi {self.interview['candidate_name']}, I am calling on behalf of TalentSkout. "
            f"I am calling to confirm, will you be available to appear for the interview on {spoken_date} "
            f"for the role of {self.interview['role']} with this company, {self.interview['company']}?"
        )

        # 1. Start continuous microphone listener in the background
        await self.vad.start()
        listen_task = asyncio.create_task(self.vad.listen_loop())

        # 2. Launch Greeting as an active cancellable task (can be interrupted at second 1!)
        self.active_agent_task = asyncio.create_task(self.speak_text(greeting, filename="live_turn_0_greeting.wav"))

        try:
            while self.is_running:
                # 3. Wait for completed candidate turn from background ear loop
                turn_type, payload = await self.user_turn_queue.get()

                # If candidate spoke while greeting/reply was playing, active_agent_task was cancelled
                if self.active_agent_task and not self.active_agent_task.done():
                    self.active_agent_task.cancel()
                self.active_agent_task = None

                # Transcribe
                if turn_type == "TEXT":
                    transcript = payload
                else:
                    t0 = time.perf_counter()
                    transcript = await async_deepgram_stt(payload)
                    stt_ms = (time.perf_counter() - t0) * 1000
                    if transcript:
                        print(f"👤 Candidate [Deepgram STT ({stt_ms:.0f}ms)]: \"{transcript}\"")
                    else:
                        continue

                if not transcript or not transcript.strip():
                    continue

                self.conversation_history.append({"role": "user", "content": transcript})

                # Check farewell
                if any(k in transcript.lower() for k in ["bye", "hang up", "thank you bye", "goodbye"]):
                    farewell = "Thank you so much! Have a wonderful day and best of luck with your interview."
                    await self.speak_text(farewell, filename="live_farewell.wav")
                    break

                self.turn_count += 1

                # Execute Brain response
                async def agent_turn():
                    async for sentence in async_stream_llm_response(transcript, self.interview, self.conversation_history):
                        self.conversation_history.append({"role": "assistant", "content": sentence})
                        await self.speak_text(sentence, filename=f"live_turn_{self.turn_count}.wav")

                self.active_agent_task = asyncio.create_task(agent_turn())
                try:
                    await self.active_agent_task
                except asyncio.CancelledError:
                    print("  [Turn cancelled due to candidate barge-in]")
                finally:
                    self.active_agent_task = None

        finally:
            self.is_running = False
            listen_task.cancel()
            await self.vad.stop()
            print("\n📴 [Call Ended] Outreach call completed.")


if __name__ == "__main__":
    interview = get_first_pending_interview()
    pipeline = RealtimeVoicePipeline(interview)
    asyncio.run(pipeline.run_conversation_loop())
