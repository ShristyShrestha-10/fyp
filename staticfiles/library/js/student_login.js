// Function to update status message
function updateStatus(message) {
    const statusMessage = document.getElementById('status-message');
    if (statusMessage) {
        statusMessage.textContent = message;
    }
}

// Function to update error message
function updateError(message) {
    const errorMessage = document.getElementById('error-message');
    if (errorMessage) {
        errorMessage.textContent = message;
        if (message) {
            errorMessage.style.display = 'block';
        } else {
            errorMessage.style.display = 'none';
        }
    }
}

// Function to update student info in the display
function updateStudentInfo(name, id) {
    const studentInfo = document.getElementById('student-info');
    if (studentInfo) {
        if (name && id) {
            studentInfo.innerHTML = `<strong>${name}</strong><br>ID: ${id}`;
            studentInfo.style.display = 'block';
        } else {
            studentInfo.style.display = 'none';
        }
    }
}

// Function to update phase indicator
function updatePhase(phase) {
    const phaseIndicator = document.getElementById('phase-indicator');
    if (phaseIndicator) {
        phaseIndicator.textContent = phase;
    }
}

// Function to handle face detection status
function handleFaceDetectionStatus(data) {
    // Handle errors
    if (data.error) {
        updateError(data.error);
    } else {
        updateError('');
    }

    // Update status message
    if (data.status) {
        updateStatus(data.status);
    }

    // Update phase information based on status
    if (data.status) {
        if (data.status.includes('Analyzing')) {
            updatePhase('Phase: Face Analysis');
        } else if (data.status.includes('recognized')) {
            updatePhase('Phase: Face Recognition');
        } else if (data.status.includes('complete')) {
            updatePhase('Phase: Complete');
        } else if (data.status.includes('Preparing')) {
            updatePhase('Phase: Preparing Face Recognition');
        }
    }

    // Update student information if available
    if (data.student_name && data.student_id) {
        updateStudentInfo(data.student_name, data.student_id);
    }

    // Handle face detection
    if (data.face_detected) {
        updateStatus('Face detected! Processing...');
    }
    
    // Check if the camera is in the 3-second waiting period
    if (data.status && data.status.includes('Preparing face recognition')) {
        const secondsMatch = data.status.match(/(\d+)s/);
        if (secondsMatch && secondsMatch[1]) {
            const secondsLeft = parseInt(secondsMatch[1]);
            updateStatus(`Preparing face recognition... ${secondsLeft}s`);
        }
    }

    // Handle face verification success
    if (data.face_saved && data.student_name && data.student_id && data.similarity_score >= 0.5) {
        updateStatus(`Welcome ${data.student_name}! Verification successful (Score: ${data.similarity_score.toFixed(2)}).`);
        updatePhase('Verification Complete');
        
        // Create a form and submit it to login the student
        const form = document.createElement('form');
        form.method = 'POST';
        form.action = LOGIN_URL;
        
        // Create and append CSRF token
        const csrfInput = document.createElement('input');
        csrfInput.type = 'hidden';
        csrfInput.name = 'csrfmiddlewaretoken';
        csrfInput.value = CSRF_TOKEN;
        form.appendChild(csrfInput);
        
        // Create and append student_id
        const studentIdInput = document.createElement('input');
        studentIdInput.type = 'hidden';
        studentIdInput.name = 'student_id';
        studentIdInput.value = data.student_id;
        form.appendChild(studentIdInput);
        
        // Append form to document body
        document.body.appendChild(form);
        
        // Submit the form after a short delay to show the welcome message
        setTimeout(() => {
            form.submit();
        }, 3000);
    } else if (data.face_saved && data.similarity_score && data.similarity_score < 0.5) {
        // Face was processed but similarity score was too low
        updateError(`Face verification failed. Similarity score (${data.similarity_score.toFixed(2)}) below threshold (0.5).`);
        updatePhase('Verification Failed');
    } else if (data.face_saved && !data.student_name) {
        // Face was processed but no match was found
        updateError('Face not recognized. Please try again or contact library staff.');
        updatePhase('Verification Failed');
    }
}

// Function to periodically check face detection status
function checkFaceDetectionStatus() {
    // Immediately return if camera is not active
    if (!window.isCameraActive) {
        console.log('Camera not active, stopping status checks');
        return;
    }

    fetch(FACE_DETECTION_STATUS_URL + "?camera_type=student_login")
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            handleFaceDetectionStatus(data);
            
            // Continue checking status if not complete and camera is still active
            if (!data.face_saved && window.isCameraActive) {
                setTimeout(checkFaceDetectionStatus, 1000);
            }
        })
        .catch(error => {
            console.error('Error checking face detection status:', error);
            updateError('Error checking status. Please try again.');
            
            // Continue checking status even if there's an error, but only if camera is still active
            if (window.isCameraActive) {
                setTimeout(checkFaceDetectionStatus, 1000);
            }
        });
}

// Initialization function
function initStudentLogin() {
    // DOM elements
    const startBtn = document.getElementById('start-btn');
    const endBtn = document.getElementById('end-btn');
    const videoFeed = document.getElementById('video-feed');
    
    // Set global state
    window.isCameraActive = false;
    window.statusCheckInterval = null;
    
    // Start checking face detection status when camera is active
    function startCamera() {
        if (window.isCameraActive) return;
        
        // Show video feed
        videoFeed.src = VIDEO_FEED_URL;
        
        // Reset messages
        updateStatus('Starting camera...');
        updateError('');
        updateStudentInfo('', '');
        updatePhase('Initializing');
        
        // Set up event listeners
        videoFeed.onload = function() {
            window.isCameraActive = true;
            startBtn.disabled = true;
            endBtn.disabled = false;
            updateStatus('Camera started. Please position your ID card in view.');
            
            // Start checking face detection status
            checkFaceDetectionStatus();
        };
        
        videoFeed.onerror = function() {
            stopCamera();
            updateError('Failed to start camera. Please try again.');
        };
    }

    // Stop checking face detection status when camera is stopped
    function stopCamera() {
        if (!window.isCameraActive) return;
        
        // Immediately update UI to indicate we're stopping
        updateStatus('Stopping camera...');
        
        // First, stop the camera on the server using an AJAX call
        fetch('/stop-scanner/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
                'X-CSRFToken': CSRF_TOKEN
            },
            body: 'camera_type=student_login'
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            console.log('Camera stopped successfully:', data);
        })
        .catch(error => {
            console.error('Error stopping camera:', error);
            updateError('Failed to stop camera properly, but interface has been reset.');
        })
        .finally(() => {
            // Always stop the video feed and reset UI regardless of server response
            videoFeed.src = '';
            window.isCameraActive = false;
            startBtn.disabled = false;
            endBtn.disabled = true;
            
            // Clear messages
            updateStatus('Camera stopped.');
            updateError('');
            updateStudentInfo('', '');
            updatePhase('');
        });
    }
    
    // Add event listeners for buttons
    if (startBtn) {
        startBtn.addEventListener('click', startCamera);
    }
    
    if (endBtn) {
        endBtn.addEventListener('click', stopCamera);
        // Initialize button states
        endBtn.disabled = true;
    }

    // Clean up on page unload
    window.addEventListener('beforeunload', function() {
        if (window.isCameraActive) {
            stopCamera();
        }
    });
}

// Initialize when DOM content is loaded
document.addEventListener('DOMContentLoaded', initStudentLogin);

// Student login function with CSRF protection
document.addEventListener('DOMContentLoaded', function() {
    // Element references
    const videoFeed = document.getElementById('video-feed');
    const startBtn = document.getElementById('start-btn');
    const endBtn = document.getElementById('end-btn');
    const statusMsg = document.getElementById('status-message');
    const errorMsg = document.getElementById('error-message');
    const studentInfo = document.getElementById('student-info');
    const phaseIndicator = document.getElementById('phase-indicator');
    
    let streamActive = false;
    let statusCheckInterval = null;
    
    // Initialize UI
    endBtn.disabled = true;
    videoFeed.src = "";
    
    // Start camera
    startBtn.addEventListener('click', function() {
        videoFeed.src = VIDEO_FEED_URL;
        streamActive = true;
        startBtn.disabled = true;
        endBtn.disabled = false;
        
        // Start status check
        statusCheckInterval = setInterval(checkStatus, 1000);
    });
    
    // End camera
    endBtn.addEventListener('click', function() {
        stopCamera();
    });
    
    // Check face detection status
    function checkStatus() {
        if (!streamActive) return;
        
        // Add CSRF token to the request
        fetch(FACE_DETECTION_STATUS_URL, {
            headers: {
                'X-CSRFToken': CSRF_TOKEN
            }
        })
        .then(response => response.json())
        .then(data => {
            // Update phase
            if (data.phase) {
                phaseIndicator.textContent = data.phase;
                phaseIndicator.style.display = 'block';
            } else {
                phaseIndicator.style.display = 'none';
            }
            
            // Update error message
            if (data.error_message) {
                errorMsg.textContent = data.error_message;
                errorMsg.style.display = 'block';
            } else {
                errorMsg.style.display = 'none';
            }
            
            // Update student info
            if (data.student_details) {
                const student = data.student_details;
                studentInfo.innerHTML = `
                    <div>Name: ${student.name}</div>
                    <div>ID: ${student.student_id}</div>
                `;
                studentInfo.style.display = 'block';
            } else {
                studentInfo.style.display = 'none';
            }
            
            // Handle login completion
            if (data.capture_complete && data.is_verified && data.student_details) {
                clearInterval(statusCheckInterval);
                
                // Show success message
                statusMsg.textContent = "Login successful! Redirecting...";
                statusMsg.style.display = 'block';
                
                // Submit the login form
                const form = document.createElement('form');
                form.method = 'POST';
                form.action = LOGIN_URL;
                
                // Add CSRF token
                const csrfField = document.createElement('input');
                csrfField.type = 'hidden';
                csrfField.name = 'csrfmiddlewaretoken';
                csrfField.value = CSRF_TOKEN;
                form.appendChild(csrfField);
                
                // Add student ID
                const studentIdField = document.createElement('input');
                studentIdField.type = 'hidden';
                studentIdField.name = 'student_id';
                studentIdField.value = data.student_details.student_id;
                form.appendChild(studentIdField);
                
                // Add verification status
                const verifiedField = document.createElement('input');
                verifiedField.type = 'hidden';
                verifiedField.name = 'is_verified';
                verifiedField.value = 'true';
                form.appendChild(verifiedField);
                
                // Add similarity score if available
                if (data.similarity_score) {
                    const scoreField = document.createElement('input');
                    scoreField.type = 'hidden';
                    scoreField.name = 'similarity_score';
                    scoreField.value = data.similarity_score;
                    form.appendChild(scoreField);
                }
                
                // Submit the form
                document.body.appendChild(form);
                form.submit();
            }
        })
        .catch(error => {
            console.error("Error checking status:", error);
        });
    }
    
    // Stop camera
    function stopCamera() {
        // Clear video source
        videoFeed.src = "";
        
        // Make API call to stop scanner
        fetch(STOP_SCANNER_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': CSRF_TOKEN
            },
            body: JSON.stringify({
                camera_type: 'student_login'
            })
        })
        .then(response => response.json())
        .then(data => {
            console.log("Camera stopped:", data);
        })
        .catch(error => {
            console.error("Error stopping camera:", error);
        })
        .finally(() => {
            // Reset UI state
            streamActive = false;
            startBtn.disabled = false;
            endBtn.disabled = true;
            
            if (statusCheckInterval) {
                clearInterval(statusCheckInterval);
                statusCheckInterval = null;
            }
            
            // Clear information displays
            studentInfo.innerHTML = "";
            phaseIndicator.innerHTML = "";
            errorMsg.innerHTML = "";
        });
    }
    
    // Ensure camera stops on page unload
    window.addEventListener('beforeunload', function() {
        if (streamActive) {
            const xhr = new XMLHttpRequest();
            xhr.open('POST', STOP_SCANNER_URL, false);
            xhr.setRequestHeader('Content-Type', 'application/json');
            xhr.setRequestHeader('X-CSRFToken', CSRF_TOKEN);
            xhr.send(JSON.stringify({ camera_type: 'student_login' }));
        }
    });
}); 