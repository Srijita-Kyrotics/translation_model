document.addEventListener('DOMContentLoaded', () => {
    const sourceText = document.getElementById('sourceText');
    const targetText = document.getElementById('targetText');
    const translateBtn = document.getElementById('translateBtn');
    const clearBtn = document.getElementById('clearBtn');
    const copyBtn = document.getElementById('copyBtn');

    // Translate Action
    translateBtn.addEventListener('click', async () => {
        const textToTranslate = sourceText.value.trim();
        
        if (!textToTranslate) {
            targetText.value = 'Please enter some English text to translate.';
            return;
        }

        // Set Loading State
        translateBtn.classList.add('loading');
        translateBtn.disabled = true;
        targetText.value = 'Translating...';

        try {
            const response = await fetch('/translate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: textToTranslate })
            });

            if (!response.ok) {
                targetText.value = `Error: Server returned status ${response.status}`;
            }

            const data = await response.json();
            
            if (data.translation) {
                targetText.value = data.translation;
            } else if (data.error) {
                targetText.value = `Error: ${data.error}`;
            } else {
                targetText.value = 'An unknown error occurred during translation.';
            }

        } catch (error) {
            targetText.value = `Network or API Error: ${error.message}`;
            console.error(error);
        } finally {
            // Remove Loading State
            translateBtn.classList.remove('loading');
            translateBtn.disabled = false;
        }
    });

    // Clear Source Action
    clearBtn.addEventListener('click', () => {
        sourceText.value = '';
        targetText.value = '';
        sourceText.focus();
    });

    // Copy Target Action
    copyBtn.addEventListener('click', async () => {
        if (!targetText.value || targetText.value === 'Translating...') return;

        try {
            await navigator.clipboard.writeText(targetText.value);
            
            // Visual feedback
            copyBtn.classList.add('success');
            const originalIcon = copyBtn.innerHTML;
            copyBtn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>';
            
            setTimeout(() => {
                copyBtn.classList.remove('success');
                copyBtn.innerHTML = originalIcon;
            }, 2000);
            
        } catch (err) {
            console.error('Failed to copy text: ', err);
        }
    });

    // Keyboard support - Translate on Ctrl+Enter
    sourceText.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            translateBtn.click();
        }
    });
});
