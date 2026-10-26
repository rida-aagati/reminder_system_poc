"""
Deepgram Aura Text-to-Speech (TTS) Standalone Module
====================================================
Easily convert any text string into natural, human-like voice audio (.wav)
using Deepgram's Aura TTS models.

Features:
  - Works with or without deepgram-sdk (uses direct REST API with zero external dependencies)
  - Supports multiple voices (Aura Asteria, Orion, Luna, Arcas, etc.)
  - Plays the generated audio on your laptop speakers (Windows native winsound)
  - Both simple synchronous function and async streaming function included.
"""

import os
import json
import asyncio
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

# Available Deepgram Aura Voices:
# - "aura-asteria-en" (Female - Warm, clear, friendly) -> Default
# - "aura-luna-en"    (Female - Calm, conversational)
# - "aura-stella-en"  (Female - Professional)
# - "aura-orion-en"   (Male - Professional, confident)
# - "aura-arcas-en"   (Male - Calm, authoritative)
# - "aura-perseus-en" (Male - Natural, approachable)

DEFAULT_VOICE = "aura-asteria-en"

# Auto-load .env file so API keys work inside VS Code automatically!
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def text_to_speech(
    text: str,
    output_path: str = "output_audio/speech.wav",
    voice: str = DEFAULT_VOICE,
    api_key: Optional[str] = None,
    play_now: bool = True,
) -> str:
    """
    Takes a string of text, sends it to Deepgram Aura TTS,
    and saves the resulting audio as a .wav file.

    Parameters:
      text (str): The text you want the AI to speak out loud.
      output_path (str): Where to save the generated audio file (.wav).
      voice (str): Deepgram voice model (default: "aura-asteria-en").
      api_key (str): Your Deepgram API key (reads from environment if None).
      play_now (bool): If True, plays the audio on your laptop speakers immediately.

    Returns:
      str: Absolute path of the saved audio file.
    """
    # 1. Resolve API Key
    key = api_key or os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        raise ValueError(
            "\n[Error] Deepgram API Key not found!\n"
            "Please set your key in PowerShell using:\n"
            "  $env:DEEPGRAM_API_KEY='your_api_key_here'\n"
            "Or pass it directly: text_to_speech(text, api_key='your_key')"
        )

    # 2. Ensure output directory exists
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n🎙️ [Deepgram TTS] Converting text to speech...")
    print(f"   Voice: {voice}")
    print(f"   Text:  \"{text}\"")

    # 3. Call Deepgram Aura REST API (requires encoding=linear16 with container=wav)
    url = f"https://api.deepgram.com/v1/speak?model={voice}&encoding=linear16&container=wav"
    payload = json.dumps({"text": text}).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Token {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            audio_data = response.read()

        # 4. Save audio to disk
        with open(target, "wb") as f:
            f.write(audio_data)

        print(f"✅ [Success] Audio saved: {target} ({len(audio_data):,} bytes)")

        # 5. Play over laptop speakers (Windows native)
        if play_now:
            try:
                import winsound
                print("🔊 [Playing Audio] Listening through laptop speakers...")
                winsound.PlaySound(str(target), winsound.SND_FILENAME)
            except Exception:
                pass

        return str(target)

    except urllib.error.HTTPError as exc:
        err_msg = exc.read().decode("utf-8")
        print(f"❌ [Deepgram API Error {exc.code}]: {err_msg}")
        raise
    except Exception as exc:
        print(f"❌ [Error generating speech]: {exc}")
        raise


async def async_text_to_speech(
    text: str,
    output_path: str = "output_audio/speech.wav",
    voice: str = DEFAULT_VOICE,
    api_key: Optional[str] = None,
    play_now: bool = False,
) -> str:
    """
    Asynchronous version of text_to_speech for asyncio event loops.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: text_to_speech(
            text=text,
            output_path=output_path,
            voice=voice,
            api_key=api_key,
            play_now=play_now,
        ),
    )


# ==============================================================================
# QUICK TEST / DEMONSTRATION
# ==============================================================================
if __name__ == "__main__":
    sample_text = (
        "Hi Rida Khan, I am calling on behalf of TalentSkout. "
        "I am calling to confirm, will you be available to appear for your "
        "Python Developer interview scheduled for tomorrow at 3 PM?"
    )

    print("================================================================================")
    print("                DEEPGRAM AURA TTS FUNCTION TEST")
    print("================================================================================")
    
    # If API key is set, it will generate real audio and play it!
    # If not set, it gives you a friendly instruction.
    if os.environ.get("DEEPGRAM_API_KEY"):
        output_file = text_to_speech(
            text=sample_text,
            output_path="output_audio/test_deepgram_voice.wav",
            voice="aura-asteria-en",
            play_now=True
        )
        print(f"\nFinished! You can play the file anytime: {output_file}")
    else:
        print("\n👉 To test with your real Deepgram key, run this in PowerShell first:")
        print("   $env:DEEPGRAM_API_KEY='your_deepgram_api_key_here'")
        print("   python deepgram_tts.py\n")
