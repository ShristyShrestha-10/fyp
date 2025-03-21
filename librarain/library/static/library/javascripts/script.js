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