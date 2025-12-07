VENV_NAME := venv_letters-to-santa
PYTHON := python3
PORT := 8000

.PHONY: serve clean venv transcribe extract-faces annotate extract-voices process-videos

# Create virtual environment if it doesn't exist
${VENV_NAME}:
	${PYTHON} -m venv ${VENV_NAME}

# Install Whisper and GPU dependencies (for GPU machines)
venv: ${VENV_NAME}
	@echo "Installing Whisper and GPU dependencies..."
	@test -f ${VENV_NAME}/bin/pip || (echo "Error: pip not found in venv. Recreating venv..." && rm -rf ${VENV_NAME} && ${PYTHON} -m venv ${VENV_NAME})
	${VENV_NAME}/bin/python -m pip install --upgrade pip
	${VENV_NAME}/bin/python -m pip install openai-whisper
	${VENV_NAME}/bin/python -m pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
	@echo "Installing face extraction dependencies..."
	${VENV_NAME}/bin/python -m pip install insightface opencv-python scikit-learn onnxruntime
	@echo "Installing voice extraction dependencies (optional)..."
	-${VENV_NAME}/bin/python -m pip install pyannote.audio || echo "Warning: pyannote.audio installation failed (optional dependency)"
	@echo "All dependencies installed successfully"

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
transcribe: ${VENV_NAME} venv
	@echo "Transcribing videos..."
	${VENV_NAME}/bin/python whisper_transcribe.py --videos-dir videos
	@echo "Transcription complete"

# Extract and cluster faces from videos
# Usage: make extract-faces VIDEOS_DIR=videos2
extract-faces: ${VENV_NAME} venv
	@VIDEOS_DIR=$${VIDEOS_DIR:-videos}; \
	echo "Extracting faces from videos in $${VIDEOS_DIR}..."; \
	${VENV_NAME}/bin/python face_extract.py --videos-dir $${VIDEOS_DIR}; \
	echo "Face extraction complete"

# Annotate videos with face detection results
# Usage: make annotate VIDEOS_DIR=videos2
annotate: ${VENV_NAME} venv
	@VIDEOS_DIR=$${VIDEOS_DIR:-videos}; \
	echo "Annotating videos from $${VIDEOS_DIR}..."; \
	${VENV_NAME}/bin/python video_annotate.py --videos-dir $${VIDEOS_DIR} --output-dir videos_annotated; \
	echo "Video annotation complete"

# Extract voices per speaker (if pyannote.audio available)
# Usage: make extract-voices VIDEOS_DIR=videos2
extract-voices: ${VENV_NAME} venv
	@VIDEOS_DIR=$${VIDEOS_DIR:-videos}; \
	echo "Extracting voices from videos in $${VIDEOS_DIR}..."; \
	${VENV_NAME}/bin/python voice_extract.py --videos-dir $${VIDEOS_DIR}; \
	echo "Voice extraction complete"

# Process videos: extract faces, annotate, and extract voices
# Usage: make process-videos VIDEOS_DIR=videos2
process-videos: ${VENV_NAME} venv
	@VIDEOS_DIR=$${VIDEOS_DIR:-videos}; \
	echo "Processing videos from directory: $${VIDEOS_DIR}"; \
	echo ""; \
	echo "Step 1: Extracting faces..."; \
	${VENV_NAME}/bin/python face_extract.py --videos-dir $${VIDEOS_DIR} || true; \
	echo ""; \
	echo "Step 2: Annotating videos..."; \
	${VENV_NAME}/bin/python video_annotate.py --videos-dir $${VIDEOS_DIR} || true; \
	echo ""; \
	echo "Step 3: Extracting voices..."; \
	${VENV_NAME}/bin/python voice_extract.py --videos-dir $${VIDEOS_DIR} || true; \
	echo ""; \
	echo "Video processing complete"

# Clean up virtual environment
clean:
	rm -rf ${VENV_NAME}
	rm -rf __pycache__
	rm -rf .pytest_cache

