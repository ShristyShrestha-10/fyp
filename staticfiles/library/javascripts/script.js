// Add hover effect for moving text
document.querySelectorAll('.moving-text').forEach(letter => {
    letter.addEventListener('mouseover', function() {
        this.style.color = '#26a69a';
        this.style.transform = 'scale(1.2) translateY(-5px)';
    });
    
    letter.addEventListener('mouseout', function() {
        this.style.color = '';
        this.style.transform = '';
    });
});
// Add animation for feature cards on scroll
document.addEventListener('DOMContentLoaded', function() {
    const featureCards = document.querySelectorAll('.feature-card');
    
    function checkScroll() {
        featureCards.forEach(card => {
            const cardPosition = card.getBoundingClientRect().top;
            const screenPosition = window.innerHeight / 1.3;
            
            if (cardPosition < screenPosition) {
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            } else {
                card.style.opacity = '0';
                card.style.transform = 'translateY(20px)';
            }
        });
    }
    
    // Set initial state
    featureCards.forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
    });
    
    // Check on scroll
    window.addEventListener('scroll', checkScroll);
    
    // Check on load
    checkScroll();
});

// Navigation active state and smooth scrolling
document.addEventListener('DOMContentLoaded', function() {
    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    // Navigation active state
    const navLinks = document.querySelectorAll('.nav-links a');
    const sections = document.querySelectorAll('section');

    window.addEventListener('scroll', () => {
        let current = '';
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            if (window.pageYOffset >= sectionTop - sectionHeight / 3) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href').includes(current)) {
                link.classList.add('active');
            }
        });
    });
});

// Login Modal Functionality
document.addEventListener('DOMContentLoaded', function() {
    const studentLoginBtn = document.getElementById('student-login');
    const adminLoginBtn = document.getElementById('admin-login');
    const loginModal = document.createElement('div');
    
    // Create modal structure
    loginModal.innerHTML = `
        <div class="login-modal">
            <div class="modal-content">
                <span class="close-modal">&times;</span>
                <h2 id="modal-title">Login</h2>
                <form id="login-form">
                    <input type="hidden" name="csrfmiddlewaretoken" value="{{ csrf_token }}">
                    <input type="text" placeholder="Username" required>
                    <input type="password" placeholder="Password" required>
                    <button type="submit">Login</button>
                </form>
            </div>
        </div>
    `;
    
    loginModal.classList.add('modal-overlay');
    document.body.appendChild(loginModal);
    
    // Hide modal by default
    loginModal.style.display = 'none';
    
    // Show student login modal
    if (studentLoginBtn) {
        studentLoginBtn.addEventListener('click', (e) => {
            e.preventDefault();
            document.getElementById('modal-title').textContent = 'Student Login';
            loginModal.style.display = 'flex';
        });
    }
    
    // Admin login button - let it work normally with page navigation
    if (adminLoginBtn) {
        adminLoginBtn.addEventListener('click', function(e) {
            // Let the default link behavior work - no preventDefault()
            return true;
        });
    }
    
    // Close modal
    loginModal.querySelector('.close-modal').addEventListener('click', () => {
        loginModal.style.display = 'none';
    });
    
    // Close modal when clicking outside
    loginModal.addEventListener('click', (e) => {
        if (e.target === loginModal) {
            loginModal.style.display = 'none';
        }
    });
    
    // Handle login form submission
    const loginForm = document.getElementById('login-form');
    if (loginForm) {
        loginForm.addEventListener('submit', (e) => {
            e.preventDefault();
            // Add your login logic here
            alert('Login functionality to be implemented');
            loginModal.style.display = 'none';
        });
    }
});

// Responsive Navigation Menu for Mobile
document.addEventListener('DOMContentLoaded', function() {
    const navLinks = document.querySelector('.nav-links');
    const hamburgerMenu = document.createElement('div');
    hamburgerMenu.classList.add('hamburger-menu');
    hamburgerMenu.innerHTML = '☰';
    
    // Only add hamburger menu on mobile
    if (window.innerWidth <= 768) {
        document.querySelector('.navigation .container').insertBefore(hamburgerMenu, navLinks);
        
        hamburgerMenu.addEventListener('click', () => {
            navLinks.classList.toggle('mobile-active');
        });
        
        // Close menu when link is clicked
        navLinks.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                navLinks.classList.remove('mobile-active');
            });
        });
    }
    
    // Adjust for responsive design
    window.addEventListener('resize', () => {
        if (window.innerWidth > 768) {
            navLinks.classList.remove('mobile-active');
        }
    });
});

// Handle Start SLMS button click and scanner functionality
document.addEventListener('DOMContentLoaded', function() {
    const startButton = document.getElementById('start-button');
    const statusMessage = document.getElementById('status-message');
    const instructions = document.getElementById('scanner-instructions');

    if (startButton) {
        startButton.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Disable button and show loading state
            startButton.disabled = true;
            startButton.textContent = 'Starting...';
            
            // Make the AJAX request
            fetch(this.href)
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        statusMessage.innerHTML = `<div class="success-message">${data.message}</div>`;
                        startButton.textContent = 'Started Successfully!';
                        instructions.style.display = 'block';
                    } else {
                        throw new Error(data.message);
                    }
                })
                .catch(error => {
                    statusMessage.innerHTML = `<div class="error-message">Error: ${error.message}</div>`;
                    startButton.textContent = 'Start SLMS';
                })
                .finally(() => {
                    startButton.disabled = false;
                });
        });
    }
});

function startSystem(event) {
    event.preventDefault();
    const button = event.target;
    const statusMessage = document.getElementById('status-message');
    const instructions = document.getElementById('scanner-instructions');

    // Disable button and show loading state
    button.disabled = true;
    button.textContent = 'Starting Scanner...';

    // Make AJAX request
    fetch(button.href)
        .then(response => {
            if (!response.ok && response.status !== 302) {
                throw new Error('Failed to start scanner');
            }
            // Show success state
            button.textContent = 'Scanner Active';
            button.classList.add('success');
            instructions.style.display = 'block';
            statusMessage.innerHTML = '<div class="success-message">ID Card Scanner started successfully</div>';
        })
        .catch(error => {
            button.textContent = 'Start ID Card Scanner';
            button.classList.add('error');
            statusMessage.innerHTML = `<div class="error-message">Error: ${error.message}</div>`;
        })
        .finally(() => {
            button.disabled = false;
        });
}

// Add event listener when document loads
document.addEventListener('DOMContentLoaded', function() {
    const startButton = document.getElementById('start-button');
    if (startButton) {
        startButton.addEventListener('click', startSystem);
    }
});