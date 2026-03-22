# Tamil Audio → English Transcript: Implementation Plan

## Overview

A Python CLI tool that converts Tamil audio/video files into English text transcripts. It uses **faster-whisper** (an optimized OpenAI Whisper engine) running locally on your NVIDIA GPU. Whisper natively supports Tamil and has a built-in `translate` task that transcribes + translates to English in a single step.

---

## Why faster-whisper?

| Feature | faster-whisper | OpenAI Whisper (vanilla) | OpenAI Whisper API |
|---|---|---|---|
| Speed | ~4x faster | Baseline | Fast (cloud) |
| VRAM usage | ~50% less | High | None (cloud) |
| Cost | Free (local) | Free (local) | ~$0.006/min |
| Tamil accuracy | Excellent | Excellent | Excellent |
| Offline capable | Yes | Yes | No |
| GPU requirement | NVIDIA 4GB+ | NVIDIA 6GB+ | None |

**Winner for your setup:** faster-whisper for local processing, with OpenAI API as an optional fallback for very long files or when you want maximum accuracy.

---

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐     ┌────────────┐
│  Input File  │ ──► │   FFmpeg     │ ──► │  faster-whisper   │ ──► │  .txt file │
│ (mp3/wav/m4a │     │ (decode/     │     │  (GPU inference)  │     │  (English  │
│  mp4/mkv/    │     │  resample)   │     │  task=translate   │     │  transcript│
│  webm/ogg)   │     │              │     │  Tamil → English  │     │  )         │
└──────────────┘     └──────────────┘     └──────────────────┘     └────────────┘
                                                   │
                                          (fallback if --use-api)
                                                   │
                                          ┌──────────────────┐
                                          │  OpenAI Whisper   │
                                          │  API (cloud)      │
                                          └──────────────────┘
```

---

## Tech Stack

| Component | Library/Tool | Purpose |
|---|---|---|
| Core engine | `faster-whisper` | Local Whisper inference on GPU via CTranslate2 |
| Audio processing | `ffmpeg` (system) | Decode any audio/video format, resample to 16kHz mono |
| CLI framework | `click` | Clean command-line interface with flags and help text |
| API fallback | `openai` | Optional OpenAI Whisper API for cloud processing |
| Config | `python-dotenv` | Load API keys from `.env` file |
| Progress | `rich` | Terminal progress bars and styled output |

---

## File Structure

```
convert-tamil-to-english-audio/
├── tamil_to_english.py       # Main CLI script (single entry point)
├── requirements.txt          # Python dependencies
├── .env.example              # Template for API key configuration
└── PLAN.md                   # This file
```

---

## Detailed Implementation Plan

### Step 1: Environment Setup

**Prerequisites to install on Windows:**

1. **Python 3.10+** — Download from python.org or use existing installation
2. **FFmpeg** — Required by faster-whisper for audio decoding
   - Install via: `winget install FFmpeg` or download from https://ffmpeg.org/download.html
   - Ensure `ffmpeg` is in your system PATH
3. **CUDA Toolkit** — Required for GPU acceleration
   - Install CUDA 12.x from https://developer.nvidia.com/cuda-downloads
   - Verify with: `nvcc --version`
4. **cuDNN** — Required by CTranslate2
   - Download from https://developer.nvidia.com/cudnn
   - Or install via: `pip install nvidia-cudnn-cu12`

### Step 2: Python Dependencies (`requirements.txt`)

```
faster-whisper>=1.0.0
click>=8.1.0
rich>=13.0.0
openai>=1.0.0
python-dotenv>=1.0.0
```

Install with:
```bash
pip install -r requirements.txt
```

### Step 3: Core Script — `tamil_to_english.py`

The script will have these main components:

#### 3a. CLI Interface (using Click)

```
Commands & Flags:
  tamil_to_english.py <input_file>          # Required: path to audio/video file

Options:
  --output, -o <path>                       # Output .txt file path (default: <input_name>_english.txt)
  --model, -m <size>                        # Model size: tiny, base, small, medium, large-v3 (default: medium)
  --timestamps / --no-timestamps            # Include timestamps in output (default: no)
  --use-api                                 # Use OpenAI Whisper API instead of local model
  --language, -l <lang>                     # Source language (default: ta for Tamil)
  --batch, -b <folder>                      # Process all audio files in a folder
  --verbose, -v                             # Show detailed processing info
```

#### 3b. Local Processing Flow (default mode)

```python
# Pseudocode
def process_local(input_file, model_size, timestamps):
    1. Validate input file exists and is a supported format
    2. Load faster-whisper model on GPU (CUDA)
       - model = WhisperModel(model_size, device="cuda", compute_type="float16")
       - First run downloads the model (~1.5GB for medium, ~3GB for large-v3)
       - Subsequent runs use cached model
    3. Run transcription with translate task
       - segments, info = model.transcribe(input_file, task="translate", language="ta")
       - task="translate" does Tamil speech → English text in one pass
    4. Collect segments and format output
       - With timestamps: "[00:00:05 → 00:00:12] Hello, how are you?"
       - Without timestamps: "Hello, how are you?"
    5. Write to output .txt file
    6. Print summary (duration, word count, processing time)
```

#### 3c. API Fallback Flow (--use-api flag)

```python
# Pseudocode
def process_api(input_file):
    1. Load OpenAI API key from .env file
    2. Open audio file
    3. Call OpenAI Whisper API with translation task
       - client.audio.translations.create(model="whisper-1", file=audio_file)
       - API handles Tamil → English translation
    4. Write response text to output .txt file
    Note: API has a 25MB file size limit per request
          For larger files, split into chunks using pydub
```

#### 3d. Batch Processing (--batch flag)

```python
# Pseudocode
def process_batch(folder_path):
    1. Scan folder for supported audio/video files
       - Extensions: .mp3, .wav, .m4a, .flac, .ogg, .mp4, .mkv, .webm, .avi
    2. Display file list and total count
    3. Process each file sequentially (to manage GPU memory)
    4. Save individual .txt files alongside originals
    5. Print batch summary (total files, total duration, total time)
```

### Step 4: Configuration (`.env.example`)

```env
# Only needed if using --use-api flag
OPENAI_API_KEY=sk-your-api-key-here
```

### Step 5: Error Handling

The script will handle these scenarios gracefully:

- **No GPU available** — Fall back to CPU mode with a warning (slower but works)
- **File not found** — Clear error message with the path that was tried
- **Unsupported format** — List supported formats in the error message
- **CUDA out of memory** — Suggest using a smaller model size or --use-api
- **API key missing** — Prompt user to set up .env file when using --use-api
- **Large files with API** — Auto-split files >25MB into chunks for API processing
- **FFmpeg not installed** — Detect and show installation instructions

---

## Model Size Guide

| Model | VRAM Needed | Speed | Tamil Accuracy | Best For |
|---|---|---|---|---|
| `tiny` | ~1 GB | Fastest | Fair | Quick drafts, testing |
| `base` | ~1 GB | Fast | Good | Short clips |
| `small` | ~2 GB | Moderate | Good | General use |
| `medium` | ~5 GB | Slower | Very Good | **Recommended default** |
| `large-v3` | ~10 GB | Slowest | Best | Maximum accuracy |

**Recommendation:** Start with `medium` model. If your GPU has <5GB free VRAM, use `small`. If accuracy isn't good enough, try `large-v3` with `--use-api` fallback.

---

## Usage Examples

```bash
# Basic: Convert a Tamil audio file to English transcript
python tamil_to_english.py recording.mp3

# Specify output path
python tamil_to_english.py recording.mp3 -o my_transcript.txt

# Use larger model for better accuracy
python tamil_to_english.py recording.mp3 -m large-v3

# Include timestamps in output
python tamil_to_english.py recording.mp3 --timestamps

# Use OpenAI API instead of local GPU
python tamil_to_english.py recording.mp3 --use-api

# Batch process all audio files in a folder
python tamil_to_english.py --batch ./my_audio_files/

# Verbose mode for debugging
python tamil_to_english.py recording.mp3 -v
```

---

## Sample Output

### Without timestamps (default):
```
Hello, today we will discuss about the new project requirements.
The client wants us to complete the first phase by next month.
We need to focus on the backend API development first.
```

### With timestamps (--timestamps):
```
[00:00:00 → 00:00:04] Hello, today we will discuss about the new project requirements.
[00:00:04 → 00:00:09] The client wants us to complete the first phase by next month.
[00:00:09 → 00:00:14] We need to focus on the backend API development first.
```

---

## Cost Breakdown

| Mode | Cost | Notes |
|---|---|---|
| Local (faster-whisper) | **Free** | One-time model download (~1.5GB for medium) |
| OpenAI API | ~$0.006/min | ~$0.36/hour of audio |

---

## Implementation Order

1. **Create `requirements.txt`** — List all dependencies
2. **Create `.env.example`** — API key template
3. **Build `tamil_to_english.py`** — Core script
   - a. CLI interface with Click
   - b. Local processing with faster-whisper
   - c. Output formatting (with/without timestamps)
   - d. API fallback mode
   - e. Batch processing mode
   - f. Error handling and user-friendly messages
   - g. Rich progress bars and summary output
4. **Test** — Verify with a sample Tamil audio file
5. **Iterate** — Tune model size based on your GPU's performance

---

## Future Enhancements (Optional)

- **SRT/VTT subtitle export** — Generate subtitle files for video use
- **Speaker diarization** — Identify different speakers in meetings (using pyannote)
- **Real-time mode** — Live microphone input for real-time translation
- **GUI wrapper** — Simple Tkinter or PyQt GUI if CLI gets tedious
- **Tamil transcript + English translation** — Output both Tamil text and English translation side by side
