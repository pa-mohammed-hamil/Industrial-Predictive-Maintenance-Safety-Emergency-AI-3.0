/**
 * INDUSTRA AI - Advanced Animation Library
 * Modern, smooth animations for industrial dashboard pages
 * 2027 Edition
 */

(function() {
    'use strict';

    // ==================== INTERSECTION OBSERVER ====================
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };

    // Fade in animation observer
    const fadeInObserver = new IntersectionObserver((entries) => {
        entries.forEach((entry, index) => {
            if (entry.isIntersecting) {
                setTimeout(() => {
                    entry.target.classList.add('animated-fade-in');
                }, index * 50); // Stagger delay
                fadeInObserver.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Slide up animation observer
    const slideUpObserver = new IntersectionObserver((entries) => {
        entries.forEach((entry, index) => {
            if (entry.isIntersecting) {
                setTimeout(() => {
                    entry.target.classList.add('animated-slide-up');
                }, index * 75);
                slideUpObserver.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Scale in animation observer
    const scaleInObserver = new IntersectionObserver((entries) => {
        entries.forEach((entry, index) => {
            if (entry.isIntersecting) {
                setTimeout(() => {
                    entry.target.classList.add('animated-scale-in');
                }, index * 60);
                scaleInObserver.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // ==================== AUTO-INITIALIZATION ====================
    function initializeAnimations() {
        // Add animation styles if not already added
        if (!document.getElementById('industra-animations')) {
            const style = document.createElement('style');
            style.id = 'industra-animations';
            style.textContent = `
                /* Animation Keyframes */
                @keyframes fadeInUp {
                    from {
                        opacity: 0;
                        transform: translateY(30px);
                    }
                    to {
                        opacity: 1;
                        transform: translateY(0);
                    }
                }

                @keyframes slideUp {
                    from {
                        opacity: 0;
                        transform: translateY(40px);
                    }
                    to {
                        opacity: 1;
                        transform: translateY(0);
                    }
                }

                @keyframes scaleInBounce {
                    0% {
                        opacity: 0;
                        transform: scale(0.8);
                    }
                    70% {
                        transform: scale(1.05);
                    }
                    100% {
                        opacity: 1;
                        transform: scale(1);
                    }
                }

                @keyframes shimmer {
                    0% {
                        background-position: -1000px 0;
                    }
                    100% {
                        background-position: 1000px 0;
                    }
                }

                @keyframes pulse {
                    0%, 100% {
                        opacity: 1;
                    }
                    50% {
                        opacity: 0.7;
                    }
                }

                @keyframes float {
                    0%, 100% {
                        transform: translateY(0px);
                    }
                    50% {
                        transform: translateY(-10px);
                    }
                }

                @keyframes rotate {
                    from {
                        transform: rotate(0deg);
                    }
                    to {
                        transform: rotate(360deg);
                    }
                }

                @keyframes glow {
                    0%, 100% {
                        box-shadow: 0 0 20px rgba(104, 85, 235, 0.3);
                    }
                    50% {
                        box-shadow: 0 0 30px rgba(104, 85, 235, 0.6);
                    }
                }

                @keyframes slideInRight {
                    from {
                        opacity: 0;
                        transform: translateX(50px);
                    }
                    to {
                        opacity: 1;
                        transform: translateX(0);
                    }
                }

                @keyframes slideInLeft {
                    from {
                        opacity: 0;
                        transform: translateX(-50px);
                    }
                    to {
                        opacity: 1;
                        transform: translateX(0);
                    }
                }

                /* Animation Classes */
                .animate-on-scroll {
                    opacity: 0;
                }

                .animated-fade-in {
                    animation: fadeInUp 0.6s ease-out forwards;
                }

                .animated-slide-up {
                    animation: slideUp 0.7s ease-out forwards;
                }

                .animated-scale-in {
                    animation: scaleInBounce 0.5s ease-out forwards;
                }

                .animated-slide-right {
                    animation: slideInRight 0.6s ease-out forwards;
                }

                .animated-slide-left {
                    animation: slideInLeft 0.6s ease-out forwards;
                }

                /* Hover Effects */
                .hover-lift {
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                }

                .hover-lift:hover {
                    transform: translateY(-8px);
                    box-shadow: 0 15px 40px rgba(36, 48, 91, 0.15);
                }

                .hover-glow:hover {
                    animation: glow 1.5s ease-in-out infinite;
                }

                .hover-scale {
                    transition: transform 0.3s ease;
                }

                .hover-scale:hover {
                    transform: scale(1.05);
                }

                .hover-brightness:hover {
                    filter: brightness(1.1);
                }

                /* Loading States */
                .shimmer-loading {
                    background: linear-gradient(
                        90deg,
                        #f0f1f7 25%,
                        #e9ebf3 50%,
                        #f0f1f7 75%
                    );
                    background-size: 1000px 100%;
                    animation: shimmer 2s infinite;
                }

                /* Smooth Transitions */
                .smooth-transition {
                    transition: all 0.3s ease-in-out;
                }

                /* Floating Animation */
                .animate-float {
                    animation: float 3s ease-in-out infinite;
                }

                /* Pulse Animation */
                .animate-pulse {
                    animation: pulse 2s ease-in-out infinite;
                }

                /* Rotate Animation */
                .animate-rotate {
                    animation: rotate 2s linear infinite;
                }

                /* Stagger Delays */
                .stagger-1 { animation-delay: 0.1s !important; }
                .stagger-2 { animation-delay: 0.2s !important; }
                .stagger-3 { animation-delay: 0.3s !important; }
                .stagger-4 { animation-delay: 0.4s !important; }
                .stagger-5 { animation-delay: 0.5s !important; }
                .stagger-6 { animation-delay: 0.6s !important; }

                /* Respect reduced motion preferences */
                @media (prefers-reduced-motion: reduce) {
                    *,
                    *::before,
                    *::after {
                        animation-duration: 0.01ms !important;
                        animation-iteration-count: 1 !important;
                        transition-duration: 0.01ms !important;
                    }
                }
            `;
            document.head.appendChild(style);
        }

        // Auto-detect elements with animation classes
        const fadeElements = document.querySelectorAll('.animate-fade-in, .surface, .metric, .report-card');
        const slideElements = document.querySelectorAll('.animate-slide-up, .hero');
        const scaleElements = document.querySelectorAll('.animate-scale-in, .metric-icon, .report-icon');

        // Add scroll animation class
        fadeElements.forEach(el => {
            el.classList.add('animate-on-scroll');
            fadeInObserver.observe(el);
        });

        slideElements.forEach(el => {
            el.classList.add('animate-on-scroll');
            slideUpObserver.observe(el);
        });

        scaleElements.forEach(el => {
            el.classList.add('animate-on-scroll');
            scaleInObserver.observe(el);
        });

        // Add hover effects to cards
        document.querySelectorAll('.surface, .metric, .report-card').forEach(el => {
            if (!el.classList.contains('hover-lift')) {
                el.classList.add('hover-lift');
            }
        });

        // Add smooth transitions to buttons
        document.querySelectorAll('.btn-premium, .btn-soft, .small-action, .top-icon').forEach(el => {
            el.classList.add('smooth-transition');
        });
    }

    // ==================== PAGE TRANSITION ====================
    function initializePageTransitions() {
        // Fade in page content on load
        document.body.style.opacity = '0';
        document.body.style.transition = 'opacity 0.5s ease-in-out';
        
        window.addEventListener('load', () => {
            setTimeout(() => {
                document.body.style.opacity = '1';
            }, 100);
        });

        // Smooth page exit
        document.querySelectorAll('a[href]:not([href^="#"])').forEach(link => {
            link.addEventListener('click', (e) => {
                if (link.getAttribute('target') !== '_blank') {
                    e.preventDefault();
                    const href = link.getAttribute('href');
                    
                    document.body.style.opacity = '0';
                    setTimeout(() => {
                        window.location.href = href;
                    }, 300);
                }
            });
        });
    }

    // ==================== COUNTER ANIMATION ====================
    function animateCounter(element, target, duration = 1000) {
        const start = 0;
        const increment = target / (duration / 16);
        let current = start;

        const timer = setInterval(() => {
            current += increment;
            if (current >= target) {
                element.textContent = target;
                clearInterval(timer);
            } else {
                element.textContent = Math.floor(current);
            }
        }, 16);
    }

    // Auto-animate counters when visible
    const counterObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const element = entry.target;
                const target = parseInt(element.textContent) || 0;
                if (target > 0) {
                    element.textContent = '0';
                    animateCounter(element, target, 1500);
                }
                counterObserver.unobserve(element);
            }
        });
    }, observerOptions);

    // Observe metric values
    document.querySelectorAll('.metric-value').forEach(el => {
        counterObserver.observe(el);
    });

    // ==================== TOAST ANIMATIONS ====================
    function enhanceToasts() {
        const toastContainer = document.querySelector('.toast-container');
        if (toastContainer) {
            const style = document.createElement('style');
            style.textContent = `
                .toast {
                    animation: slideInRight 0.4s ease-out;
                }
                
                .toast.hiding {
                    animation: slideInRight 0.4s ease-out reverse;
                }
            `;
            document.head.appendChild(style);
        }
    }

    // ==================== MODAL ANIMATIONS ====================
    function enhanceModals() {
        document.querySelectorAll('.modal').forEach(modal => {
            modal.addEventListener('show.bs.modal', () => {
                const dialog = modal.querySelector('.modal-dialog');
                dialog.style.animation = 'scaleInBounce 0.4s ease-out';
            });
        });
    }

    // ==================== LOADING STATES ====================
    window.IndustraUI = window.IndustraUI || {};
    
    window.IndustraUI.showLoading = function(element) {
        const originalContent = element.innerHTML;
        element.setAttribute('data-original-content', originalContent);
        element.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Loading...';
        element.disabled = true;
    };

    window.IndustraUI.hideLoading = function(element) {
        const originalContent = element.getAttribute('data-original-content');
        if (originalContent) {
            element.innerHTML = originalContent;
            element.disabled = false;
        }
    };

    window.IndustraUI.createSkeleton = function(element, lines = 3) {
        element.innerHTML = '';
        for (let i = 0; i < lines; i++) {
            const skeleton = document.createElement('div');
            skeleton.className = 'shimmer-loading rounded mb-2';
            skeleton.style.height = '20px';
            skeleton.style.width = `${Math.random() * 30 + 70}%`;
            element.appendChild(skeleton);
        }
    };

    // ==================== RIPPLE EFFECT ====================
    function addRippleEffect() {
        document.querySelectorAll('.btn-premium, .btn-soft').forEach(button => {
            button.addEventListener('click', function(e) {
                const ripple = document.createElement('span');
                const rect = this.getBoundingClientRect();
                const size = Math.max(rect.width, rect.height);
                const x = e.clientX - rect.left - size / 2;
                const y = e.clientY - rect.top - size / 2;

                ripple.style.width = ripple.style.height = size + 'px';
                ripple.style.left = x + 'px';
                ripple.style.top = y + 'px';
                ripple.style.position = 'absolute';
                ripple.style.borderRadius = '50%';
                ripple.style.background = 'rgba(255, 255, 255, 0.5)';
                ripple.style.transform = 'scale(0)';
                ripple.style.animation = 'ripple 0.6s ease-out';
                ripple.style.pointerEvents = 'none';

                this.style.position = 'relative';
                this.style.overflow = 'hidden';
                this.appendChild(ripple);

                setTimeout(() => ripple.remove(), 600);
            });
        });

        // Add ripple animation CSS
        const style = document.createElement('style');
        style.textContent = `
            @keyframes ripple {
                to {
                    transform: scale(4);
                    opacity: 0;
                }
            }
        `;
        document.head.appendChild(style);
    }

    // ==================== PARALLAX EFFECT ====================
    function initializeParallax() {
        const parallaxElements = document.querySelectorAll('[data-parallax]');
        
        if (parallaxElements.length > 0) {
            window.addEventListener('scroll', () => {
                const scrolled = window.pageYOffset;
                
                parallaxElements.forEach(el => {
                    const speed = el.getAttribute('data-parallax') || 0.5;
                    el.style.transform = `translateY(${scrolled * speed}px)`;
                });
            });
        }
    }

    // ==================== PROGRESS BAR ANIMATION ====================
    const progressObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const progressBar = entry.target.querySelector('.progress-fill, .metric-fill');
                if (progressBar) {
                    const width = progressBar.style.width || progressBar.getAttribute('data-width');
                    progressBar.style.width = '0%';
                    setTimeout(() => {
                        progressBar.style.transition = 'width 1s ease-out';
                        progressBar.style.width = width;
                    }, 100);
                }
                progressObserver.unobserve(entry.target);
            }
        });
    }, observerOptions);

    document.querySelectorAll('.progress-bar, .metric-bar').forEach(el => {
        progressObserver.observe(el);
    });

    // ==================== INITIALIZE ALL ====================
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            initializeAnimations();
            enhanceToasts();
            enhanceModals();
            addRippleEffect();
            initializeParallax();
        });
    } else {
        initializeAnimations();
        enhanceToasts();
        enhanceModals();
        addRippleEffect();
        initializeParallax();
    }

    // Reinitialize on dynamic content changes
    window.IndustraUI = window.IndustraUI || {};
    window.IndustraUI.reinitializeAnimations = initializeAnimations;

    console.log('✨ INDUSTRA AI Animations loaded');
})();
