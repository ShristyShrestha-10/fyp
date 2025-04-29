// This script handles the redirection after successful login
document.addEventListener('DOMContentLoaded', function() {
    // If this script is loaded via AJAX after a successful login
    if (window.loginSuccessData) {
        handleLoginSuccess(window.loginSuccessData);
    }
    
    // Check if there's a JSON response in the page (direct rendering of the response)
    checkForJsonInPage();
    
    // Function to handle form submission
    function setupLoginForm() {
        const loginForm = document.getElementById('login-form');
        if (loginForm) {
            loginForm.addEventListener('submit', function(e) {
                e.preventDefault();
                
                const formData = new FormData(loginForm);
                
                fetch(loginForm.action, {
                    method: 'POST',
                    body: formData,
                    headers: {
                        'X-Requested-With': 'XMLHttpRequest'
                    }
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        handleLoginSuccess(data);
                    } else {
                        // Display error message
                        if (document.getElementById('error-message')) {
                            document.getElementById('error-message').textContent = data.message;
                            document.getElementById('error-message').style.display = 'block';
                        }
                    }
                })
                .catch(error => {
                    console.error('Error during login:', error);
                    if (document.getElementById('error-message')) {
                        document.getElementById('error-message').textContent = 'An error occurred during login. Please try again.';
                        document.getElementById('error-message').style.display = 'block';
                    }
                });
            });
        }
    }
    
    // Function to handle login success
    function handleLoginSuccess(data) {
        console.log('Login successful. Redirecting to:', data.redirect_url);
        
        // Show success message if available
        if (document.getElementById('success-message')) {
            document.getElementById('success-message').textContent = data.message || 'Login successful! Redirecting...';
            document.getElementById('success-message').style.display = 'block';
        }
        
        // Redirect to main page
        setTimeout(function() {
            window.location.href = data.redirect_url;
        }, 500);
    }
    
    // Function to check for JSON in the page content
    function checkForJsonInPage() {
        const content = document.body.textContent || '';
        if (content.includes('"status": "success"') && content.includes('"redirect_url":')) {
            try {
                const jsonRegex = /\{[\s\S]*?"status"[\s\S]*?:[\s\S]*?"success"[\s\S]*?\}/;
                const match = content.match(jsonRegex);
                if (match && match[0]) {
                    let jsonText = match[0];
                    // Clean up any leading/trailing text
                    jsonText = jsonText.replace(/^[^{]+/, '').replace(/[^}]+$/, '');
                    
                    console.log('Found JSON in page content:', jsonText);
                    try {
                        const data = JSON.parse(jsonText);
                        if (data.status === 'success' && data.redirect_url) {
                            console.log('Extracted valid JSON with redirect URL. Redirecting...');
                            handleLoginSuccess(data);
                        }
                    } catch (parseError) {
                        console.error('Error parsing extracted JSON:', parseError);
                    }
                }
            } catch (e) {
                console.error('Error extracting JSON from page:', e);
            }
        }
    }
    
    // Set up forms
    setupLoginForm();
}); 