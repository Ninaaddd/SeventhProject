document.addEventListener('DOMContentLoaded', function () {
    const speakNameButton = document.getElementById('speakNameButton');
    const nameInput = document.getElementById('nameInput');

    const speakNumberButton = document.getElementById('speakNumberButton');
    const numberInput = document.getElementById('numberInput');

    const submitButton = document.getElementById('submitButton');
    const loginButton = document.getElementById('loginButton'); // Assuming the login button has an ID

    function startSpeechRecognition(inputElement, isPhoneNumber = false) {
        const recognition = new webkitSpeechRecognition() || new SpeechRecognition();

        recognition.onresult = function (event) {
            const result = event.results[0][0].transcript;
            const cleanedResult = isPhoneNumber ? result.replace(/\D/g, '') : result.replace(/[\d\s]/g, '');

            inputElement.value = cleanedResult;

            if (isPhoneNumber) {
                // Validate that the cleaned result has exactly 10 digits
                if (!/^\d{10}$/.test(cleanedResult)) {
                    speakErrorMessage(isPhoneNumber);
                }
            }
        };

        recognition.onerror = function (event) {
            console.error('Speech recognition error:', event.error);
            recognition.stop();
        };

        recognition.onend = function () {
            console.log('Speech recognition ended.');
        };

        recognition.start();
    }

    speakNameButton.addEventListener('click', function () {
        startSpeechRecognition(nameInput);
    });

    speakNumberButton.addEventListener('click', function () {
        startSpeechRecognition(numberInput, true); // Pass true to indicate it's a phone number input
    });

    submitButton.addEventListener('mouseover', function (event) {
        const nameInputValue = nameInput.value;
        const phNumberInputValue = numberInput.value;

        // Validate name (text-only)
        if (/[^a-zA-Z]/.test(nameInputValue)) {
            speakErrorMessage(false);
            event.preventDefault(); // Prevent form submission
        }

        // Validate phone number (exactly 10 digits)
        if (!/^\d{10}$/.test(phNumberInputValue)) {
            speakErrorMessage(true);
            event.preventDefault(); // Prevent form submission
        }
    });

    speakNameButton.addEventListener('mouseover', function () {
        speakButtonText(speakNameButton);
    });

    speakNumberButton.addEventListener('mouseover', function () {
        speakButtonText(speakNumberButton);
    });

    submitButton.addEventListener('mouseover', function (event) {
        const phNumberInput = numberInput.value;
        if (!/^\d{10}$/.test(phNumberInput)) {
            speakErrorMessage(true);
            event.preventDefault(); // Prevent form submission
        } else {
            speakButtonText(submitButton);
        }
    });

    loginButton.addEventListener('mouseover', function (event) {
        speakButtonText(loginButton);
    });

    function speakButtonText(buttonElement) {
        const buttonText = buttonElement.innerText;
        const speech = new SpeechSynthesisUtterance(buttonText);
        speechSynthesis.speak(speech);
    }

    function speakErrorMessage(isPhoneNumber) {
        const errorMessage = isPhoneNumber
            ? "You have provided a wrong phone number. Please give only a 10-digit mobile number."
            : "You have provided wrong information. Please give only text for the name.";
        const errorSpeech = new SpeechSynthesisUtterance(errorMessage);
        speechSynthesis.speak(errorSpeech);
    }
});