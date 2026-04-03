document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    const solveBtn = document.getElementById('solve-btn');
    const loading = document.getElementById('loading');
    const resultContainer = document.getElementById('result-container');
    const formulaLatex = document.getElementById('formula-latex');
    const formulaRendered = document.getElementById('formula-rendered');
    const solvedText = document.getElementById('solved-text');
    const methodBadge = document.getElementById('method-badge');
    const filenameText = document.getElementById('filename-text');

    let selectedFile = null;

    // --- Drag & Drop Handlers ---
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    ['dragleave', 'dragend'].forEach(type => {
        dropZone.addEventListener(type, () => {
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            previewContainer.classList.remove('hidden');
            solveBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        selectedFile = null;
        fileInput.value = '';
        previewContainer.classList.add('hidden');
        solveBtn.disabled = true;
        resultContainer.classList.add('hidden');
    });

    // --- API Request ---
    solveBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // Reset UI
        resultContainer.classList.add('hidden');
        loading.classList.remove('hidden');
        solveBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/solve', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.status === 'success') {
                const data = result.data;
                
                // Set data
                formulaLatex.textContent = data.formula;
                
                // Prepare MathJax input
                // Wrap in $$ for block display
                formulaRendered.innerHTML = `\\[ ${data.formula} \\]`;
                
                solvedText.textContent = data.solved;
                methodBadge.textContent = data.method;
                filenameText.textContent = result.filename;

                // Show results
                loading.classList.add('hidden');
                resultContainer.classList.remove('hidden');
                
                // Trigger MathJax re-render
                if (window.MathJax) {
                    MathJax.typesetPromise([formulaRendered]).catch(err => console.error(err));
                }

            } else {
                throw new Error(result.error || 'Prediction failed');
            }
        } catch (error) {
            console.error('Error:', error);
            alert(`Calculation Failed: ${error.message}`);
            loading.classList.add('hidden');
        } finally {
            solveBtn.disabled = false;
        }
    });
});
