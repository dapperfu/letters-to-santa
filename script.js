let mediaRecorder;
let recordedChunks = [];
let stream = null;

const preview = document.getElementById('preview');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const status = document.getElementById('status');

/**
 * Display a status message to the user.
 * 
 * Parameters
 * ----------
 * message : string
 *     The message to display.
 * type : string
 *     The type of message: 'info', 'success', 'error', or 'recording'.
 */
function showStatus(message, type = 'info') {
    status.textContent = message;
    status.className = `status-message show ${type}`;
    
    if (type === 'success') {
        setTimeout(() => {
            status.classList.remove('show');
        }, 3000);
    }
}

/**
 * Request access to the user's webcam.
 * 
 * Returns
 * -------
 * Promise<MediaStream>
 *     A promise that resolves to the media stream.
 */
async function getWebcam() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: true
        });
        preview.srcObject = stream;
        showStatus('Camera ready! Click to start recording.', 'info');
        return stream;
    } catch (error) {
        console.error('Error accessing webcam:', error);
        showStatus('Unable to access camera. Please check permissions.', 'error');
        startBtn.disabled = true;
        throw error;
    }
}

/**
 * Start recording video from the webcam.
 */
function startRecording() {
    if (!stream) {
        showStatus('Camera not available. Please refresh the page.', 'error');
        return;
    }

    recordedChunks = [];
    
    const options = {
        mimeType: 'video/webm;codecs=vp9,opus'
    };
    
    // Fallback to default if codec not supported
    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
        options.mimeType = 'video/webm';
    }
    
    try {
        mediaRecorder = new MediaRecorder(stream, options);
        
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                recordedChunks.push(event.data);
            }
        };
        
        mediaRecorder.onstop = () => {
            uploadVideo();
        };
        
        mediaRecorder.start();
        showStatus('Recording...', 'recording');
        startBtn.style.display = 'none';
        stopBtn.style.display = 'inline-block';
    } catch (error) {
        console.error('Error starting recording:', error);
        showStatus('Error starting recording. Please try again.', 'error');
    }
}

/**
 * Stop recording video.
 */
function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        showStatus('Processing your message...', 'info');
        stopBtn.style.display = 'none';
        startBtn.style.display = 'inline-block';
        startBtn.disabled = true;
    }
}

/**
 * Upload the recorded video to the server.
 */
async function uploadVideo() {
    const blob = new Blob(recordedChunks, { type: 'video/webm' });
    
    const formData = new FormData();
    formData.append('video', blob, 'recording.webm');
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok && result.success) {
            showStatus('Message sent to Santa! 🎅', 'success');
            // Reset after a delay
            setTimeout(() => {
                recordedChunks = [];
                startBtn.disabled = false;
            }, 3000);
        } else {
            throw new Error(result.error || 'Upload failed');
        }
    } catch (error) {
        console.error('Error uploading video:', error);
        showStatus('Error sending message. Please try again.', 'error');
        startBtn.disabled = false;
    }
}

/**
 * Stop all media tracks to release the camera.
 */
function stopWebcam() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
}

// Event listeners
startBtn.addEventListener('click', startRecording);
stopBtn.addEventListener('click', stopRecording);

// Initialize webcam on page load
getWebcam().catch(error => {
    console.error('Failed to initialize webcam:', error);
});

// Clean up on page unload
window.addEventListener('beforeunload', stopWebcam);

