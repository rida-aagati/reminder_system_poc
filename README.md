# TalentSkout Automated Reminder & Asynchronous Voice Agent System

A production-ready, fully asynchronous streaming voice agent and two-stage automated reminder architecture built for **TalentSkout**.

---

## 🎯 Architecture & Improvements

| PoC Area | Architecture Implementation |
| :--- | :--- |
| **Concurrency** | **100% Asynchronous (`asyncio`)**: Non-blocking event loop across STT, LLM Streaming, and TTS. |
| **LLM Delivery** | **True Streaming LLM**: Streams token-by-token, chunked into sentences for instant speech delivery with full tool support (`confirm_attendance`, `get_interview_context`). |
| **Barge-In Interruption** | **Active Task Cancellation**: Turn execution wrapped in `asyncio.Task`; when user interruption is detected, `task.cancel()` immediately aborts active LLM streaming and halts audio output. |
| **Voice Synthesis** | **Real Deepgram Aura TTS (`aura-asteria-en`)**: Generates 16kHz linear PCM audio saved to `output_audio/` and played over speakers. |
| **Microphone & VAD** | **Smart Live VAD**: Measures ambient room noise dynamically, filters hiss/clicks with 3-chunk debouncing (~90ms), and uses an echo lock to prevent self-interruption. |
| **Security Guardrails** | **Code-Level Input & Output Filtering**: Blocks prompt injections and exfiltration attempts before reaching the LLM; redacts recruiter/panelist personal contact numbers from speech. |

---

## 🔄 Two-Stage Reminder Timing Rules

1. **Stage 1 (AI Screener / 48-Hour Valid Link)**:
   - **Email Reminder**: Dispatched $\le 12\text{ hours}$ before expiration with note: *"NOTE: You will be reached out by a phone call if you are unable to complete the interview and need to be reminded."*
   - **Outbound Voice Call**: Triggered $\le 2\text{ hours}$ before expiration.
   - **Skip Rule**: Automatically skipped if `status == "completed"` or `attendance_confirmed == true`.

2. **Stage 2 (Panel Interview with Human Panelists)**:
   - **Email Reminder**: Dispatched $\le 24\text{ hours}$ before scheduled slot.
   - **Outbound Voice Call**: Triggered $\le 4\text{ hours}$ before scheduled slot.
   - **Skip Rule**: Automatically skipped if `status == "completed"` or `attendance_confirmed == true`.

---

## 📁 File Structure

```text
Reminder System/
├── .env                        # Local environment variables (DEEPGRAM_API_KEY)
├── .env.example                # Example configuration template
├── db.json                     # Candidate database (AI Screener & Panel interview records)
├── reset_db.py                 # Resets db.json to initial pending states for repeatable testing
├── deepgram_tts.py             # Standalone Deepgram Aura TTS function (sync + async) with speaker playback
├── generate_sample_audio.py    # Generates 16kHz WAV speech files for the 5 test scenarios
├── realtime_voice_agent.py     # Live microphone streaming agent with ambient calibration & barge-in
├── main.py                     # Full asynchronous test suite (5 distinct WAV inputs + barge-in demo)
├── voice_agent.py              # Modular copy of main asynchronous engine
├── reminder_system.py          # Unified system with scheduler & async pipeline
├── requirements.txt            # Python dependencies (sounddevice, numpy, anthropic, deepgram-sdk)
├── sample_audio/               # 5 distinct 16kHz PCM WAV test audio files:
│   ├── turn_1_confirm.wav            (Attendance confirmation)
│   ├── turn_2_panelist_query.wav     (Panelist query)
│   ├── turn_3_timeline_query.wav     (Link expiration query)
│   ├── turn_4_injection_attack.wav   (Prompt injection attack test)
│   └── turn_5_offtopic.wav           (Off-topic car insurance query test)
└── output_audio/               # Generated Deepgram Aura WAV audio output files:
    ├── turn_0_intro.wav
    ├── turn_1_agent.wav
    ├── ...
    └── test_deepgram_voice.wav
```

---

## 🚀 How to Run

### Step 1: Install Dependencies
Open your PowerShell or VS Code terminal in the project directory:
```powershell
python -m pip install sounddevice numpy
```

*(Optional: `anthropic` and `deepgram-sdk` can be installed if desired, but all scripts include direct REST API fallbacks with zero required external dependencies).*

### Step 2: Configure Environment Variables
Ensure `.env` exists in the project root with your Deepgram API key:
```env
DEEPGRAM_API_KEY=your_deepgram_api_key_here
```
*(All scripts automatically load `.env` on launch, working seamlessly in VS Code and PowerShell).*

---

### Step 3: Run the Scripts

#### 1. Test Standalone Deepgram Aura TTS (Instant Voice Playback)
```powershell
python deepgram_tts.py
```
Converts text into human-quality voice audio, saves `output_audio/test_deepgram_voice.wav`, and plays it immediately through your laptop speakers!

#### 2. Run the Full 5-Turn Asynchronous Audio Test Suite
```powershell
python main.py
```
Executes:
- **Two-Stage Reminder Dispatch**: Evaluates all candidate deadlines and sends mock emails.
- **Mode A (5 Audio Inputs)**: Transcribes all 5 WAV files with Deepgram Nova-3 STT, streams through the LLM engine with tool execution, and synthesizes speech via Deepgram Aura TTS into `output_audio/`.
- **Mode B (Active Barge-In Demo)**: Simulates mid-stream speech interruption and proves `task.cancel()` aborts the active pipeline immediately.

#### 3. Run the Live Microphone Voice Agent
```powershell
python realtime_voice_agent.py
```
1. Automatically measures your room's ambient noise floor for 0.8s to prevent false triggers.
2. The agent greets you through your speakers using Deepgram Aura TTS.
3. Speak into your microphone to confirm attendance, ask about panelists, or test guardrails.
4. If you speak while the agent is talking, **barge-in** instantly silences the speaker and redirects.

#### 4. Reset the Database Anytime
```powershell
python reset_db.py
```
Restores `db.json` back to initial pending states so all test triggers and calls can be re-run cleanly.

---

## 🛡️ Guardrails & Limitations

- **Greeting Format**: Strictly adheres to:
  > *"Hi {name}, I am calling on behalf of TalentSkout. I am calling to confirm, will you be available to appear for the interview on {date} for the role of {role} with this company, {company}?"*
- **Attendance Confirmation**: Upon candidate confirmation, executes `confirm_attendance` $\rightarrow$ updates `status: "confirmed"`, `attendance_confirmed: true`, `call_reminder_sent: true`.
- **Input Guardrail**: Code-level regex intercepts prompt injections (`"discard all previous instructions"`, `"DAN mode"`) and confidential data exfiltration before invoking the LLM.
- **Output Guardrail**: Redacts private recruiter/panelist phone numbers (`[Contact through TalentSkout Portal]`) and strips markdown/bullets before TTS.
- **Off-Topic Scope Limitation**: Declines irrelevant inquiries (car insurance, weather, loans) with:
  > *"I'm here only to assist you with questions regarding your upcoming TalentSkout interview. Are you planning to appear for the interview?"*
