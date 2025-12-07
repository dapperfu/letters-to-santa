VENV_NAME := venv_vibe_letters_to_santa
PYTHON := python3
PORT := 8000

.PHONY: serve clean

# Create virtual environment if it doesn't exist
${VENV_NAME}:
	${PYTHON} -m venv ${VENV_NAME}

# Serve the application
serve: ${VENV_NAME}
	@echo "Starting server on port ${PORT}..."
	@${VENV_NAME}/bin/python server.py ${PORT} & \
	SERVER_PID=$$!; \
	echo "Server started with PID: $$SERVER_PID"; \
	sleep 2; \
	echo "Launching Firefox in kiosk mode..."; \
	firefox --kiosk http://localhost:${PORT} & \
	FIREFOX_PID=$$!; \
	echo "Firefox started with PID: $$FIREFOX_PID"; \
	echo ""; \
	echo "Server and browser are running."; \
	echo "Press Ctrl+C to stop both."; \
	trap "kill $$SERVER_PID $$FIREFOX_PID 2>/dev/null; exit" INT TERM; \
	wait $$SERVER_PID

# Clean up virtual environment
clean:
	rm -rf ${VENV_NAME}
	rm -rf __pycache__
	rm -rf .pytest_cache

