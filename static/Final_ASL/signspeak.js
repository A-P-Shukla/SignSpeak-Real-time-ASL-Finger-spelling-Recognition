const textArea = document.getElementById('prediction-result');

textArea.value = ""; // Initialize the text area with an empty string

setInterval(() => {  // Send requests to the backend every 100ms (adjust as needed)
    fetch('/predict', { method: 'POST' })  // Send a POST request to /predict
        .then(response => {
            if (!response.ok) {  // Check for network/server errors
                console.error("Network response was not ok: ", response.status);
                throw new Error("Network response was not ok"); // Re-throw for catch block
            }
            return response.json(); // Parse the JSON response
        })
        .then(data => {
            if (data.error) { //check for error key in response.
                console.error('Prediction Error:', data.error);
                // Handle error (e.g., display error in the text area)
            } else {
                textArea.value = data.predicted_string;  // Update the text area
            }
        })
        .catch(error => {
            console.error('Fetch Error:', error);
            // Handle error (e.g., display error in the text area)
        });
}, 100);  // Adjust the interval (100ms) if needed