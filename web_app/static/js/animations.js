// Animation Utilities
class AnimationManager {
    constructor() {
        this.observer = null;
        this.init();
    }

    init() {
        this.setupScrollAnimations();
        this.setupHoverEffects();
        this.setupLoadingAnimations();
    }

    setupScrollAnimations() {
        // Intersection Observer for scroll animations
        this.observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-in');
                }
            });
        }, {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        });

        // Observe elements with animation classes
        document.querySelectorAll('.feature-card, .step-item, .metric-card').forEach(el => {
            this.observer.observe(el);
        });
    }

    setupHoverEffects() {
        // Add hover effects to cards
        document.querySelectorAll('.card').forEach(card => {
            card.addEventListener('mouseenter', () => {
                this.animateCardHover(card, true);
            });
            
            card.addEventListener('mouseleave', () => {
                this.animateCardHover(card, false);
            });
        });

        // Button hover effects
        document.querySelectorAll('.btn-primary').forEach(btn => {
            btn.addEventListener('mouseenter', () => {
                this.animateButtonHover(btn, true);
            });
            
            btn.addEventListener('mouseleave', () => {
                this.animateButtonHover(btn, false);
            });
        });
    }

    setupLoadingAnimations() {
        // Loading spinner variations
        this.createPulseAnimation();
        this.createWaveAnimation();
    }

    animateCardHover(card, isHovering) {
        if (isHovering) {
            card.style.transform = 'translateY(-8px)';
            card.style.boxShadow = 'var(--shadow-xl)';
        } else {
            card.style.transform = 'translateY(0)';
            card.style.boxShadow = 'var(--shadow-lg)';
        }
    }

    animateButtonHover(btn, isHovering) {
        if (isHovering) {
            btn.style.transform = 'translateY(-2px) scale(1.05)';
        } else {
            btn.style.transform = 'translateY(0) scale(1)';
        }
    }

    createPulseAnimation() {
        const style = document.createElement('style');
        style.textContent = `
            @keyframes pulse {
                0% { transform: scale(1); opacity: 1; }
                50% { transform: scale(1.05); opacity: 0.8; }
                100% { transform: scale(1); opacity: 1; }
            }
            
            .pulse-animation {
                animation: pulse 2s infinite;
            }
        `;
        document.head.appendChild(style);
    }

    createWaveAnimation() {
        const style = document.createElement('style');
        style.textContent = `
            @keyframes wave {
                0% { transform: translateX(-100%); }
                100% { transform: translateX(100%); }
            }
            
            .wave-effect {
                position: relative;
                overflow: hidden;
            }
            
            .wave-effect::after {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(
                    90deg,
                    transparent,
                    rgba(255, 255, 255, 0.2),
                    transparent
                );
                animation: wave 1.5s infinite;
            }
        `;
        document.head.appendChild(style);
    }

    // Typewriter effect for text
    typewriterEffect(element, text, speed = 50) {
        let i = 0;
        element.textContent = '';
        
        function type() {
            if (i < text.length) {
                element.textContent += text.charAt(i);
                i++;
                setTimeout(type, speed);
            }
        }
        
        type();
    }

    // Count up animation for numbers
    countUp(element, target, duration = 2000) {
        const start = 0;
        const increment = target / (duration / 16);
        let current = start;
        
        const timer = setInterval(() => {
            current += increment;
            if (current >= target) {
                element.textContent = target.toFixed(1);
                clearInterval(timer);
            } else {
                element.textContent = current.toFixed(1);
            }
        }, 16);
    }

    // Fade in elements sequentially
    fadeInSequence(elements, delay = 200) {
        elements.forEach((el, index) => {
            setTimeout(() => {
                el.style.opacity = '0';
                el.style.transform = 'translateY(20px)';
                el.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
                
                setTimeout(() => {
                    el.style.opacity = '1';
                    el.style.transform = 'translateY(0)';
                }, 10);
            }, index * delay);
        });
    }

    // Shake animation for errors
    shake(element) {
        element.style.animation = 'none';
        setTimeout(() => {
            element.style.animation = 'shake 0.5s ease';
        }, 10);
        
        // Add shake animation if not exists
        if (!document.querySelector('#shake-animation')) {
            const style = document.createElement('style');
            style.id = 'shake-animation';
            style.textContent = `
                @keyframes shake {
                    0%, 100% { transform: translateX(0); }
                    10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
                    20%, 40%, 60%, 80% { transform: translateX(5px); }
                }
            `;
            document.head.appendChild(style);
        }
    }
}

// Initialize animations
function initAnimations() {
    window.animations = new AnimationManager();
    
    // Add animation classes to elements
    document.addEventListener('DOMContentLoaded', () => {
        const animatedElements = document.querySelectorAll('.feature-card, .step-item');
        if (window.animations && window.animations.fadeInSequence) {
            window.animations.fadeInSequence(Array.from(animatedElements));
        }
    });
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { AnimationManager, initAnimations };
}