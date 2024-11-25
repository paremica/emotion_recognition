# Real-Time Emotion-Based Car Recommendation System

This project demonstrates a **Real-Time Emotion Detection System** that evaluates user emotions through a webcam feed, identifies their emotional state, and recommends car options dynamically. The system uses a **Deep Learning Model**, **FastAPI** for backend services, and a **WebSocket connection** for real-time updates. The objective is to determine whether the client likes a car based on their emotional reaction. If negative emotions are detected (e.g., angry, fear, disgust), the system switches the car displayed.

---

## Features
1. **Emotion Detection**: Real-time detection of emotions like happy, angry, surprise, fear, etc.
2. **Car Recommendation**: Dynamically changes the car displayed based on user emotions.
3. **Deep Learning Model**:
   - Trained on the [Face Expression Recognition Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset).
   - Achieves emotion classification using a convolutional neural network (CNN).
4. **WebSocket Integration**: Provides real-time updates of emotion detection and confidence scores.
5. **FastAPI Backend**: Handles WebSocket communication and serves the frontend files.
6. **MongoDB**: Stores preprocessed training data as binary image files.
7. **Dockerized Application**: Fully containerized using Docker for platform-independent deployment.

---

## Math Behind Emotion Detection

The emotion detection is modeled as a **multi-class classification problem**, where each emotion is treated as a separate class. The model minimizes the **categorical cross-entropy loss**:

L = -(1/N) * Σ[i=1 to N] Σ[j=1 to C] y_ij * log(ŷ_ij)

Where:
- N: Total number of samples.
- C: Number of classes (e.g., happy, angry, etc.).
- y_ij: True label for class j of sample i.
- ŷ_ij: Predicted probability for class j of sample i.

The emotion with the highest predicted probability is selected as the output.

---

## System Workflow
### Phase 1: Data Preparation
1. **Dataset**: Used the [Face Expression Recognition Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset).
2. **MongoDB Storage**: Converted the images into binary format and stored them in a MongoDB database.
    ```python
    img = Image.open(io.BytesIO(image_binary)).convert("L")
    img = img.resize((48, 48))
    images.append(img_to_array(img))
    labels.append(label)
    ```

3. **Data Splitting**: Split the data into training (80%) and testing (20%) sets using `train_test_split`.

### Phase 2: Model Training
1. **Architecture**: The CNN includes:
   - 3 Convolutional layers with 128, 256, and 512 filters, each followed by MaxPooling and Dropout.
   - Fully connected layers with 512 and 256 neurons.
   - Output layer with 7 neurons (one per emotion class) and a Softmax activation.

    ```python
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    ```

2. **Compilation**: Optimized using Adam optimizer and categorical cross-entropy loss.

3. **Training**: The model was trained for 50 epochs with a batch size of 128:
    ```python
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=128)
    ```

4. **Saving Artifacts**:
   - Trained model: `emotion_model.h5`
   - Label encoder: `label_encoder_classes.npy`

---

### Phase 3: Backend Implementation
1. **FastAPI**:
   - Serves the frontend (`index.html`, CSS, and JavaScript).
   - Provides a WebSocket endpoint for real-time emotion detection:
     ```python
     @app.websocket("/real-time-emotion")
     ```

2. **WebSocket**: Reads frames from the webcam, processes them using OpenCV, and predicts emotions:
   ```python
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   predictions = model.predict(face)
   emotion_index = np.argmax(predictions)
   ```

---

### Phase 4: Frontend Logic
1. **Webcam Access**:
   - Utilizes the browser's `getUserMedia` API to access the webcam.
   ```javascript
   navigator.mediaDevices.getUserMedia({ video: true })
   ```

2. **Dynamic Car Recommendation**:
   - Based on the detected emotion:
     ```javascript
     if (emotion === "angry" || emotion === "fear" || emotion === "disgust") {
         setTimeout(switchCar, 2000);
     }
     ```

3. **Real-Time Updates**:
   - Updates emotion and confidence scores dynamically:
     ```javascript
     emotionDisplay.textContent = `Emotion: ${emotion}`;
     confidenceDisplay.textContent = `Confidence: ${(confidence * 100).toFixed(2)}%`;
     ```

---

## Dockerization
1. **Dockerfile**:
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt /app/requirements.txt
   RUN pip install -r requirements.txt
   COPY . /app
   CMD ["uvicorn", "main1:app", "--host", "0.0.0.0", "--port", "8080"]
   ```

2. **Build and Run**:
   ```bash
   docker build -t emotion-recognition-app .
   docker run -p 8080:8080 emotion-recognition-app
   ```

---

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/paremica/emotion_recognition.git
   ```
2. Build and run the Docker container:
   ```bash
   docker build -t emotion-recognition-app .
   docker run -p 8080:8080 emotion-recognition-app
   ```
3. Open `http://localhost:8080` in your browser.
4. Allow webcam permissions and view the real-time emotion-based car recommendation system in action!

---

