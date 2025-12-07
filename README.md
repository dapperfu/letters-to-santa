# Letters to Santa 🎅

A single-page web application that allows children to record video messages to Santa using their webcam. Videos are saved with timestamps and can be transcribed using OpenAI Whisper on GPU machines.

## Why Can't We Send Video Requests to Santa?

**Child:** "Why can't we send video requests to Santa?"

**Developer:** "Well, you see... Santa's workshop runs on a very old email system from the 1980s. It only supports text-based messages, not video attachments. The elves tried to upgrade it once, but the reindeer network kept crashing."

**Child:** "But my friend said Santa has a YouTube channel!"

**Developer:** "Ah yes, that's Santa's *public* channel for watching kids' wish lists. But the actual letter delivery system? Still running on punch cards and carrier pigeons. Very secure, you know."

**Child:** "Can't we just upload it to the cloud?"

**Developer:** "Santa's cloud is actually a real cloud - he stores everything in the sky. But video files are too heavy for clouds to carry. That's why we save them here first, and then... well, we're still working on the cloud-to-sky transfer protocol."

**Child:** "So what do we do with these videos?"

**Developer:** "We save them with timestamps, transcribe them with AI (because even Santa needs help reading sometimes), and then... we figure it out. Maybe we'll print them out and send them via traditional mail? Or train carrier pigeons to carry USB drives? The possibilities are endless!"

**Child:** "That doesn't make any sense."

**Developer:** "Exactly! Welcome to software development! 🎄"

## System Requirements

- **Python 3.x** - For the web server
- **Chromium browser** - Required for kiosk mode (not Firefox)
- **Webcam** - For recording video messages
- **GPU (optional)** - For Whisper transcription (CUDA/ROCm support)

## Installation

1. Clone this repository
2. Run `make serve` to start the server and launch Chromium in kiosk mode

## Usage

### Running the Application

```bash
make serve
```

This will:
- Create a Python virtual environment if needed
- Start the HTTP server on port 8000
- Launch Chromium in full-screen kiosk mode

### Exiting Kiosk Mode

**Press `Alt+F4`** to exit Chromium kiosk mode and stop the server.

### Recording Videos

1. Click "Send Message to Santa" to start recording
2. Speak your message to Santa
3. Click "Stop Recording" when finished
4. Your video will be saved with a timestamp in the `videos/` directory

### Transcribing Videos (GPU Machines)

For machines with GPU support:

1. Install Whisper dependencies:
   ```bash
   make venv
   ```

2. Transcribe all videos:
   ```bash
   make transcribe
   ```

This will generate:
- SRT subtitle files: `video_YYYYMMDD_HHMMSS.srt`
- Plain text files: `video_YYYYMMDD_HHMMSS.txt`

## Project Structure

```
.
├── index.html          # Main web page
├── style.css           # Styling
├── script.js           # Webcam and recording logic
├── server.py           # HTTP server
├── whisper_transcribe.py  # Whisper transcription module
├── Makefile            # Build and serve commands
├── videos/             # Directory for recorded videos
└── README.md           # This file

```

## License

MIT License - See LICENSE file for details.

## Notes

- Videos are saved in WebM format
- Transcripts are generated using OpenAI Whisper
- The application runs in kiosk mode for a distraction-free experience
- Make sure Chromium is installed before running `make serve`

