// DOM Elements
const titleInput = document.getElementById('title');
const descriptionInput = document.getElementById('description');
const inputDescInput = document.getElementById('input-description');
const outputDescInput = document.getElementById('output-description');
const predictBtn = document.getElementById('predict-btn');
const loadSampleBtn = document.getElementById('load-sample');
const clearResultsBtn = document.getElementById('clear-results');
const resultCard = document.getElementById('result-card');
const placeholder = document.getElementById('placeholder');
const loading = document.getElementById('loading');

// Character counter for description
descriptionInput.addEventListener('input', function() {
    const count = this.value.length;
    document.getElementById('desc-count').textContent = count;
});

// Load Sample Problem
loadSampleBtn.addEventListener('click', async function() {
    try {
        const response = await fetch('/sample');
        const data = await response.json();
        const samples = data.samples;
        
        if (samples && samples.length > 0) {
            const sample = samples[Math.floor(Math.random() * samples.length)];
            titleInput.value = sample.title;
            descriptionInput.value = sample.description;
            inputDescInput.value = sample.input_description;
            outputDescInput.value = sample.output_description;
            
            // Update character count
            document.getElementById('desc-count').textContent = sample.description.length;
            
            showNotification('Sample problem loaded!', 'success');
        }
    } catch (error) {
        showNotification('Failed to load sample', 'error');
        console.error(error);
    }
});

// Clear Results
clearResultsBtn.addEventListener('click', function() {
    resultCard.classList.add('hidden');
    placeholder.classList.remove('hidden');
    showNotification('Results cleared', 'info');
});

// Make Prediction
// ... existing code ...

// Make Prediction
predictBtn.addEventListener('click', async function() {
    // Validate inputs
    if (!titleInput.value.trim() || !descriptionInput.value.trim()) {
        showNotification('Please enter problem title and description', 'error');
        return;
    }
    
    if (descriptionInput.value.trim().length < 10) {
        showNotification('Problem description should be at least 10 characters', 'warning');
        return;
    }
    
    // Show loading
    predictBtn.disabled = true;
    loading.classList.remove('hidden');
    resultCard.classList.add('hidden');
    
    try {
        // Prepare data
        const data = {
            title: titleInput.value.trim(),
            description: descriptionInput.value.trim(),
            input_description: inputDescInput.value.trim(),
            output_description: outputDescInput.value.trim()
        };
        
        // Make API call
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayPrediction(result.prediction);
            showNotification('Prediction successful!', 'success');
        } else {
            // Show detailed error message
            let errorMsg = result.error || 'Prediction failed';
            if (result.instructions) {
                errorMsg += `. ${result.instructions}`;
            }
            throw new Error(errorMsg);
        }
        
    } catch (error) {
        showNotification(`Error: ${error.message}`, 'error');
        console.error('Prediction error details:', error);
        
        // Show instructions for fixing
        if (error.message.includes('Models not loaded')) {
            showNotification('Please train the models first. Run: cd ml_model && python train.py', 'warning');
        }
    } finally {
        // Hide loading
        predictBtn.disabled = false;
        loading.classList.add('hidden');
    }
});

// ... rest of the code ...
// Display Prediction Results
function displayPrediction(prediction) {
    // Hide placeholder, show result card
    placeholder.classList.add('hidden');
    resultCard.classList.remove('hidden');
    
    // Update difficulty class
    const difficultyBadge = document.getElementById('difficulty-badge');
    const classResult = document.getElementById('class-result');
    
    classResult.textContent = prediction.problem_class;
    difficultyBadge.className = `difficulty-badge ${prediction.problem_class.toLowerCase()}`;
    
    // Update score
    const scoreValue = document.getElementById('score-value');
    const scaleFill = document.getElementById('scale-fill');
    
    scoreValue.textContent = prediction.problem_score.toFixed(2);
    scaleFill.style.width = `${(prediction.problem_score / 10) * 100}%`;
    
    // Update confidence
    const classConfidence = document.getElementById('class-confidence');
    const scoreConfidence = document.getElementById('score-confidence');
    const classConfidenceBar = document.getElementById('class-confidence-bar');
    const scoreConfidenceBar = document.getElementById('score-confidence-bar');
    
    classConfidence.textContent = `${Math.round(prediction.class_confidence * 100)}%`;
    scoreConfidence.textContent = `${Math.round(prediction.score_confidence * 100)}%`;
    classConfidenceBar.style.width = `${prediction.class_confidence * 100}%`;
    scoreConfidenceBar.style.width = `${prediction.score_confidence * 100}%`;
    
    // Add animation
    animateValue(scoreValue, 0, prediction.problem_score, 1000);
}

// Animate number counter
function animateValue(element, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const value = progress * (end - start) + start;
        element.textContent = value.toFixed(2);
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

// Show notification
function showNotification(message, type = 'info') {
    // Remove existing notification
    const existing = document.querySelector('.notification');
    if (existing) existing.remove();
    
    // Create notification
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <i class="fas fa-${getIcon(type)}"></i>
        <span>${message}</span>
    `;
    
    // Style notification
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        background: ${getColor(type)};
        color: white;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        z-index: 1000;
        animation: slideInRight 0.3s ease;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    `;
    
    document.body.appendChild(notification);
    
    // Auto remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

function getIcon(type) {
    switch(type) {
        case 'success': return 'check-circle';
        case 'error': return 'exclamation-circle';
        case 'warning': return 'exclamation-triangle';
        default: return 'info-circle';
    }
}

function getColor(type) {
    switch(type) {
        case 'success': return 'linear-gradient(135deg, #10b981, #34d399)';
        case 'error': return 'linear-gradient(135deg, #ef4444, #f87171)';
        case 'warning': return 'linear-gradient(135deg, #f59e0b, #fbbf24)';
        default: return 'linear-gradient(135deg, #3b82f6, #6366f1)';
    }
}

// Add CSS for slide animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Keyboard shortcut: Ctrl+Enter to predict
document.addEventListener('keydown', function(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        predictBtn.click();
    }
});

// Initialize with welcome message
window.addEventListener('load', function() {
    setTimeout(() => {
        showNotification('Welcome to AutoJudge! Enter a programming problem to predict difficulty.', 'info');
    }, 1000);
});