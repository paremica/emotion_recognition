const video = document.getElementById("video");
const carImage = document.getElementById("car-image");
const priceTag = document.querySelector(".price-tag");
const emotionDisplay = document.getElementById("emotion-display");
const confidenceDisplay = document.getElementById("confidence-display");

// Array of car images and prices
const cars = [
    { src: "https://i.ibb.co/6wSb4Br/car1.jpg", price: "$400,000" },
    { src: "https://i.ibb.co/GVym4Fz/car2.jpg", price: "$450,000" },
    { src: "https://i.ibb.co/bXjhQ9y/car3.jpg", price: "$500,000" },
    { src: "https://i.ibb.co/J2R8JQk/car4.jpg", price: "$550,000" },
    { src: "https://i.ibb.co/D8pGK0n/car5.jpg", price: "$600,000" },
    { src: "https://i.ibb.co/ZfjxLK0/car6.jpg", price: "$400,000" }
];

let currentCarIndex = 0;

// Function to switch car image and price
function switchCar() {
    currentCarIndex = (currentCarIndex + 1) % cars.length;
    carImage.src = cars[currentCarIndex].src;
    priceTag.textContent = cars[currentCarIndex].price;
}

// Function to change background color based on emotion
function changeBackgroundColor(emotion) {
    let color;

    switch (emotion) {
        case "happy":
            color = "#008000"; // Green
            break;
        case "surprise":
            color = "#90EE90"; // Light green
            break;
        case "neutral":
            color = "#FFFF00"; // Yellow
            break;
        case "disgust":
            color = "#FFA07A"; // Light red
            break;
        case "angry":
            color = "#FF0000"; // Red
            break;
        case "fear":
            color = "#FFA500"; // Orange
            break;
        default:
            color = "#f5f5f5"; // Default background
    }

    document.body.style.backgroundColor = color;
}

// WebSocket configuration
const socket = new WebSocket("ws://127.0.0.1:8080/real-time-emotion");

socket.onopen = () => {
    console.log("WebSocket connection established.");
};

socket.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.emotions && data.emotions.length > 0) {
        // Process the first detected emotion
        const emotionData = data.emotions[0]; // Take the first detected emotion
        const { emotion, confidence } = emotionData;

        // Update the emotion and confidence display
        emotionDisplay.textContent = `Emotion: ${emotion}`;
        confidenceDisplay.textContent = `Confidence: ${(confidence * 100).toFixed(2)}%`;

        // Change background color based on emotion
        changeBackgroundColor(emotion);

        // Switch car image for positive or neutral emotions
        if (emotion === "angry" || emotion === "fear" || emotion === "disgust") {
            setTimeout(switchCar, 2000);
        }
    } else if (data.error) {
        console.error("WebSocket Error:", data.error);
        alert(`Error: ${data.error}`);
    }
};

socket.onerror = (error) => {
    console.error("WebSocket connection error:", error);
};

socket.onclose = () => {
    console.log("WebSocket connection closed.");
};

// Access the camera
if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
            video.srcObject = stream;
            video.play();
        })
        .catch((err) => {
            console.error("Error accessing webcam:", err);
            alert(`Could not access your webcam. Please check your permissions: ${err.message}`);
        });
} else {
    alert("Your browser does not support webcam access. Please update to a modern browser.");
}

