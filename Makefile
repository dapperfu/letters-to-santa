VENV_NAME := venv_vibe_letters_to_santa
PYTHON := python3
PORT := 8000

.PHONY: serve clean venv transcribe

# Create virtual environment if it doesn't exist
${VENV_NAME}:
	${PYTHON} -m venv ${VENV_NAME}

# Install Whisper and GPU dependencies (for GPU machines)
venv: ${VENV_NAME}
	@echo "Installing Whisper and GPU dependencies..."
	@test -f ${VENV_NAME}/bin/pip || (echo "Error: pip not found in venv. Recreating venv..." && rm -rf ${VENV_NAME} && ${PYTHON} -m venv ${VENV_NAME})
	${VENV_NAME}/bin/python -m pip install --upgrade pip
	${VENV_NAME}/bin/python -m pip install openai-whisper
	${VENV_NAME}/bin/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	@echo "Whisper dependencies installed successfully"

# Serve the application
serve: ${VENV_NAME}
	@echo "Starting server on port ${PORT}..."
	@${VENV_NAME}/bin/python server.py ${PORT} & \
	SERVER_PID=$$!; \
	echo "Server started with PID: $$SERVER_PID"; \
	sleep 2; \
	echo "Launching Chromium in kiosk mode..."; \
	chromium --kiosk http://localhost:${PORT} & \
	CHROMIUM_PID=$$!; \
	echo "Chromium started with PID: $$CHROMIUM_PID"; \
	echo ""; \
	echo "Server and browser are running."; \
	echo "Press Ctrl+C to stop both."; \
	trap "kill $$SERVER_PID $$CHROMIUM_PID 2>/dev/null; exit" INT TERM; \
	wait $$SERVER_PID

# Transcribe all videos in videos/ directory
transcribe: venv
	@echo "Transcribing videos..."
	${VENV_NAME}/bin/python whisper_transcribe.py --videos-dir videos
	@echo "Transcription complete"

# Clean up virtual environment
clean:
	rm -rf ${VENV_NAME}
	rm -rf __pycache__
	rm -rf .pytest_cache

