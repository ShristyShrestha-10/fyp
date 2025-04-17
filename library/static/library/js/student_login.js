// Function to update status message
function updateStatus(message) {
    const statusMessage = document.getElementById('status-message');
    statusMessage.textContent = message;
}

// Function to update error message
function updateError(message) {
    const errorMessage = document.getElementById('error-message');
    errorMessage.textContent = message;
}

// Function to handle face detection status
function handleFaceDetectionStatus(data) {
    if (data.error) {
        updateError(data.error);
        return;
    }

    if (data.status) {
        updateStatus(data.status);
    }

    if (data.face_detected) {
        updateStatus('Face detected! Processing...');
    }

    if (data.face_saved) {
        updateStatus('Face saved successfully!');
        // Redirect to verification page after a short delay
        setTimeout(() => {
            window.location.href = "{% url 'verify_face' %}";
        }, 2000);
    }
}

// Function to periodically check face detection status
function checkFaceDetectionStatus() {
    if (!isCameraActive) return;

    fetch("{% url 'get_face_detection_status' %}")
        .then(response => response.json())
        .then(data => {
            handleFaceDetectionStatus(data);
            // Continue checking status
            setTimeout(checkFaceDetectionStatus, 1000);
        })
        .catch(error => {
            console.error('Error checking face detection status:', error);
            updateError('Error checking face detection status');
            // Continue checking status even if there's an error
            setTimeout(checkFaceDetectionStatus, 1000);
        });
}

// Start checking face detection status when camera is active
function startCamera() {
    if (isCameraActive) return;
    
    // Show video feed
    videoFeed.src = "{% url 'video_feed' camera_type='student_login' %}";
    
    // Set up event listeners
    videoFeed.onload = function() {
        isCameraActive = true;
        startBtn.disabled = true;
        endBtn.disabled = false;
        // Start checking face detection status
        checkFaceDetectionStatus();
    };
    
    videoFeed.onerror = function() {
        stopCamera();
        window.location.href = "{% url 'home' %}";
    };
}

// Stop checking face detection status when camera is stopped
function stopCamera() {
    if (!isCameraActive) return;

    // Stop the video feed
    videoFeed.src = '';
    isCameraActive = false;
    startBtn.disabled = false;
    endBtn.disabled = true;
    statusMessage.textContent = '';
    errorMessage.textContent = '';

    // Redirect to home page
    window.location.href = "{% url 'home' %}";
} 