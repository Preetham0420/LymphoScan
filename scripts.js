document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("fileInput");
    const fileNameDisplay = document.getElementById("fileName");
    const previewImage = document.getElementById("previewImage");
    const resetButton = document.getElementById("resetButton");
    const predictButton = document.getElementById("predictButton");
    const predictionOutput = document.getElementById("predictionOutput");

    fileInput.addEventListener("change", function () {
        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            const fileType = file.type;

            fileNameDisplay.textContent = file.name;

            const reader = new FileReader();
            reader.onload = function (e) {
                previewImage.src = e.target.result;
                previewImage.style.display = "block"; // Ensure preview is shown
            };

            if (fileType.startsWith("image/")) {
                reader.readAsDataURL(file);
            } else {
                console.error("Unsupported file format");
            }

            resetButton.classList.remove("hidden"); // Show reset button
        } else {
            fileNameDisplay.textContent = "No file chosen";
            previewImage.src = "";
            previewImage.style.display = "none"; // Hide if no file
            resetButton.classList.add("hidden");
        }
    });

    //  Reset Button Functionality (Now Opens File Explorer)
    resetButton.addEventListener("click", function () {
        fileInput.value = ""; // Clear file input
        fileInput.click(); // Open file explorer for new selection
    });

    //  Handle Prediction (FIXED!)
    predictButton.addEventListener("click", function (event) {
        event.preventDefault(); // Prevent form submission from reloading page

        if (!fileInput.files.length) {
            predictionOutput.innerHTML = "❌ Please upload an image before predicting.";
            predictionOutput.style.display = "block";
            return;
        }

        const formData = new FormData();
        formData.append("file", fileInput.files[0]);

        fetch("/predict", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            predictionOutput.innerHTML = `✅ Prediction: ${data.prediction}`;
            predictionOutput.style.display = "block";
        })
        .catch(error => {
            predictionOutput.innerHTML = '❌ Error in prediction.';
            predictionOutput.style.display = "block";
        });
    });
});
