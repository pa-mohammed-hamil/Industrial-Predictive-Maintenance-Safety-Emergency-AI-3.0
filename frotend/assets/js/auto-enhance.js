/**
 * INDUSTRA AI - Auto Page Enhancer
 * Automatically adds animations and enhancements to all pages
 * Just include this script and it does everything!
 */

(function() {
    'use strict';

    console.log('🎨 INDUSTRA AI Auto-Enhancer initializing...');

    // ==================== ADD ENHANCED STYLES ====================
    function addEnhancedStyles() {
        // Check if styles are already added
        if (document.getElementById('industra-enhanced-styles')) {
            return;
        }

        const link = document.createElement('link');
        link.id = 'industra-enhanced-styles';
        link.rel = 'stylesheet';
        link.href = '/assets/css/enhanced-industra.css';
        document.head.appendChild(link);

        console.log('✅ Enhanced styles loaded');
    }

    // ==================== ADD ANIMATIONS LIBRARY ====================
    function addAnimationsLibrary() {
        // Check if animations are already added
        if (document.getElementById('industra-animations-script')) {
            return;
        }

        const script = document.createElement('script');
        script.id = 'industra-animations-script';
        script.src = '/assets/js/animations.js';
        document.head.appendChild(script);

        console.log('✅ Animations library loaded');
    }

    // ==================== ENHANCE EXISTING ELEMENTS ====================
    function enhanceElements() {
        // Add smooth scrolling to all internal links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                const targetId = this.getAttribute('href');
                if (targetId === '#') return;
                
                const targetElement = document.querySelector(targetId);
                if (targetElement) {
                    e.preventDefault();
                    targetElement.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });

        // Add loading states to forms
        document.querySelectorAll('form').forEach(form => {
            form.addEventListener('submit', function(e) {
                const submitBtn = form.querySelector('[type="submit"]');
                if (submitBtn && !submitBtn.hasAttribute('data-no-loading')) {
                    const originalText = submitBtn.innerHTML;
                    submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing...';
                    submitBtn.disabled = true;

                    // Reset after 3 seconds (in case form doesn't navigate away)
                    setTimeout(() => {
                        submitBtn.innerHTML = originalText;
                        submitBtn.disabled = false;
                    }, 3000);
                }
            });
        });

        // Add tooltips to elements with title attribute
        document.querySelectorAll('[title]').forEach(el => {
            if (!el.hasAttribute('data-bs-toggle')) {
                el.setAttribute('data-bs-toggle', 'tooltip');
            }
        });

        // Initialize Bootstrap tooltips if available
        if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
            const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
            tooltipTriggerList.map(function (tooltipTriggerEl) {
                return new bootstrap.Tooltip(tooltipTriggerEl);
            });
        }

        // Add focus-visible polyfill for better keyboard navigation
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Tab') {
                document.body.classList.add('user-is-tabbing');
            }
        });

        document.addEventListener('mousedown', function() {
            document.body.classList.remove('user-is-tabbing');
        });

        console.log('✅ Elements enhanced');
    }

    // ==================== ADD NUMBER COUNTING ANIMATION ====================
    function animateNumbers() {
        const observerOptions = {
            root: null,
            rootMargin: '0px',
            threshold: 0.1
        };

        const numberObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const element = entry.target;
                    const finalValue = element.textContent.replace(/[^0-9.-]/g, '');
                    const numericValue = parseFloat(finalValue);

                    if (!isNaN(numericValue) && !element.hasAttribute('data-animated')) {
                        element.setAttribute('data-animated', 'true');
                        animateValue(element, 0, numericValue, 1500);
                    }

                    numberObserver.unobserve(element);
                }
            });
        }, observerOptions);

        // Observe metric values
        document.querySelectorAll('.metric-value, [data-animate-number]').forEach(el => {
            numberObserver.observe(el);
        });

        console.log('✅ Number animations ready');
    }

    function animateValue(element, start, end, duration) {
        const startTime = performance.now();
        const suffix = element.textContent.replace(/[0-9.-]/g, '');

        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Easing function (ease-out-cubic)
            const easeOut = 1 - Math.pow(1 - progress, 3);
            const current = start + (end - start) * easeOut;

            // Format the number
            let displayValue;
            if (end % 1 === 0) {
                displayValue = Math.floor(current);
            } else {
                displayValue = current.toFixed(1);
            }

            element.textContent = displayValue + suffix;

            if (progress < 1) {
                requestAnimationFrame(update);
            } else {
                element.textContent = end + suffix;
            }
        }

        requestAnimationFrame(update);
    }

    // ==================== ADD PARTICLE EFFECTS (OPTIONAL) ====================
    function addParticleEffects() {
        const hero = document.querySelector('.hero');
        if (!hero || hero.querySelector('.particles')) {
            return;
        }

        const particlesContainer = document.createElement('div');
        particlesContainer.className = 'particles';
        particlesContainer.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            pointer-events: none;
            z-index: 0;
        `;

        // Create floating particles
        for (let i = 0; i < 15; i++) {
            const particle = document.createElement('div');
            particle.style.cssText = `
                position: absolute;
                width: ${Math.random() * 4 + 2}px;
                height: ${Math.random() * 4 + 2}px;
                background: rgba(255, 255, 255, 0.5);
                border-radius: 50%;
                top: ${Math.random() * 100}%;
                left: ${Math.random() * 100}%;
                animation: float-particle ${Math.random() * 10 + 10}s infinite ease-in-out;
                animation-delay: ${Math.random() * -20}s;
            `;
            particlesContainer.appendChild(particle);
        }

        // Add CSS animation
        if (!document.getElementById('particle-animation')) {
            const style = document.createElement('style');
            style.id = 'particle-animation';
            style.textContent = `
                @keyframes float-particle {
                    0%, 100% {
                        transform: translate(0, 0) scale(1);
                        opacity: 0;
                    }
                    10% {
                        opacity: 0.6;
                    }
                    50% {
                        transform: translate(${Math.random() * 100 - 50}px, ${Math.random() * -100}px) scale(1.2);
                        opacity: 0.8;
                    }
                    90% {
                        opacity: 0.6;
                    }
                }
            `;
            document.head.appendChild(style);
        }

        hero.insertBefore(particlesContainer, hero.firstChild);
        console.log('✅ Particle effects added');
    }

    // ==================== ADD DYNAMIC HOVER EFFECTS ====================
    function addDynamicHoverEffects() {
        // Add 3D tilt effect to cards
        document.querySelectorAll('.surface, .metric, .report-card').forEach(card => {
            card.addEventListener('mousemove', function(e) {
                if (window.innerWidth < 768) return; // Skip on mobile

                const rect = this.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;

                const centerX = rect.width / 2;
                const centerY = rect.height / 2;

                const deltaX = (x - centerX) / centerX;
                const deltaY = (y - centerY) / centerY;

                const tiltX = deltaY * 5; // Max 5 degrees
                const tiltY = -deltaX * 5;

                this.style.transform = `perspective(1000px) rotateX(${tiltX}deg) rotateY(${tiltY}deg) translateY(-8px) scale(1.01)`;
            });

            card.addEventListener('mouseleave', function() {
                this.style.transform = '';
            });
        });

        console.log('✅ Dynamic hover effects added');
    }

    // ==================== PERFORMANCE MONITORING ====================
    function monitorPerformance() {
        if ('performance' in window && 'measure' in window.performance) {
            const loadTime = performance.now();
            console.log(`⚡ Page enhanced in ${loadTime.toFixed(2)}ms`);

            // Report Largest Contentful Paint
            if ('PerformanceObserver' in window) {
                try {
                    const lcpObserver = new PerformanceObserver((list) => {
                        const entries = list.getEntries();
                        const lastEntry = entries[entries.length - 1];
                        console.log(`📊 LCP: ${lastEntry.renderTime.toFixed(2)}ms`);
                    });
                    lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] });
                } catch (e) {
                    // Observer not supported
                }
            }
        }
    }

    // ==================== ACCESSIBILITY ENHANCEMENTS ====================
    function enhanceAccessibility() {
        // Add skip to main content link
        if (!document.querySelector('.skip-to-main')) {
            const skipLink = document.createElement('a');
            skipLink.className = 'skip-to-main';
            skipLink.href = '#main';
            skipLink.textContent = 'Skip to main content';
            skipLink.style.cssText = `
                position: absolute;
                left: -9999px;
                z-index: 9999;
                padding: 1em;
                background: var(--purple);
                color: white;
                text-decoration: none;
                font-weight: bold;
                border-radius: 0 0 8px 0;
            `;
            skipLink.addEventListener('focus', function() {
                this.style.left = '0';
            });
            skipLink.addEventListener('blur', function() {
                this.style.left = '-9999px';
            });
            document.body.insertBefore(skipLink, document.body.firstChild);
        }

        // Ensure all images have alt text
        document.querySelectorAll('img:not([alt])').forEach(img => {
            img.setAttribute('alt', 'Decorative image');
            console.warn('Image missing alt text:', img.src);
        });

        // Add ARIA labels to icons-only buttons
        document.querySelectorAll('button, a').forEach(el => {
            if (!el.textContent.trim() && !el.getAttribute('aria-label')) {
                const icon = el.querySelector('i[class*="bi-"]');
                if (icon) {
                    const iconClass = Array.from(icon.classList).find(c => c.startsWith('bi-'));
                    const label = iconClass ? iconClass.replace('bi-', '').replace(/-/g, ' ') : 'Button';
                    el.setAttribute('aria-label', label);
                }
            }
        });

        console.log('✅ Accessibility enhanced');
    }

    // ==================== ERROR HANDLING ====================
    function setupErrorHandling() {
        window.addEventListener('error', function(e) {
            console.error('🚨 Page error:', e.error);
            // Could send to analytics here
        });

        window.addEventListener('unhandledrejection', function(e) {
            console.error('🚨 Unhandled promise rejection:', e.reason);
        });
    }

    // ==================== INITIALIZE ALL ====================
    function initialize() {
        try {
            addEnhancedStyles();
            addAnimationsLibrary();
            enhanceElements();
            animateNumbers();
            addParticleEffects();
            addDynamicHoverEffects();
            enhanceAccessibility();
            setupErrorHandling();
            monitorPerformance();

            console.log('🎉 INDUSTRA AI Auto-Enhancer initialized successfully!');
        } catch (error) {
            console.error('❌ Error initializing enhancer:', error);
        }
    }

    // Run when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initialize);
    } else {
        initialize();
    }

    // Expose reinit function for dynamic content
    window.IndustraEnhancer = {
        reinitialize: initialize,
        animateNumbers: animateNumbers,
        enhanceElements: enhanceElements
    };

})();
