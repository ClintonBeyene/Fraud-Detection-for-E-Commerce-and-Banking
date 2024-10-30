document.addEventListener('DOMContentLoaded', function() {
    const fraudCheckForm = document.getElementById('fraudCheckForm');
    const resultDiv = document.getElementById('result');
    const resultAlert = resultDiv.querySelector('.alert');

    fraudCheckForm.addEventListener('submit', function(e) {
        e.preventDefault();

        // Get form data
        const formData = {
            amount: document.getElementById('amount').value,
            country: document.getElementById('country').value,
            merchantType: document.getElementById('merchantType').value
        };

        // Show loading state
        const submitButton = fraudCheckForm.querySelector('button[type="submit"]');
        const originalButtonText = submitButton.innerHTML;
        submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Checking...';
        submitButton.disabled = true;

        // Make API request
        fetch('/check_fraud', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrf_token // Make sure csrf_token is defined
            },
            body: JSON.stringify(formData)
        })
        .then(response => response.json())
        .then(data => {
            // Show result
            resultDiv.style.display = 'block';
            if (data.prediction === 1) {
                resultAlert.className = 'alert alert-danger';
                resultAlert.textContent = 'This transaction appears to be fraudulent!';
            } else {
                resultAlert.className = 'alert alert-success';
                resultAlert.textContent = 'This transaction appears to be legitimate.';
            }
        })
        .catch(error => {
            resultDiv.style.display = 'block';
            resultAlert.className = 'alert alert-warning';
            resultAlert.textContent = 'An error occurred while checking the transaction.';
            console.error('Error:', error);
        })
        .finally(() => {
            // Reset button state
            submitButton.innerHTML = originalButtonText;
            submitButton.disabled = false;
        });
    });

    // Form validation
    const inputs = fraudCheckForm.querySelectorAll('input, select');
    inputs.forEach(input => {
        input.addEventListener('input', function() {
            if (this.hasAttribute('required')) {
                if (this.value) {
                    this.classList.remove('is-invalid');
                } else {
                    this.classList.add('is-invalid');
                }
            }
        });
    });
});