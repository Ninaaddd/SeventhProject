document.addEventListener('DOMContentLoaded', function () {
    const speakNumberButton = document.getElementById('speakNumberButton');
    const numberInput = document.getElementById('numberInput');
    const signupButton = document.getElementById('signupButton');

    speakNumberButton.addEventListener('click', function () {
        startSpeechRecognition(numberInput);
    });

    speakNumberButton.addEventListener('mouseover', function () {
        speakButtonText(speakNumberButton);
    });

    signupButton.addEventListener('click', function () {
        window.location.href = signupButton.getAttribute('href');
    });

    function startSpeechRecognition(inputElement) {
        const recognition = new webkitSpeechRecognition() || new SpeechRecognition();

        recognition.onresult = function (event) {
            const result = event.results[0][0].transcript;
            inputElement.value = result;
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

    function speakButtonText(buttonElement) {
        const buttonText = buttonElement.innerText;
        const speech = new SpeechSynthesisUtterance(buttonText);
        speechSynthesis.speak(speech);
    }
});