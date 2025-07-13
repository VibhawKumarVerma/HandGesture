# HandGesture Solver

# ✋🤖 Handwritten Math Solver using AI (Gesture-Controlled)

This project is a gesture-based math-solving application that lets users draw math problems in the air using their fingers. The drawing is captured using a webcam, interpreted by AI (Google Gemini), and the result is displayed on screen.

---

## 📌 Features

- ✍️ Draw with index finger in the air
- 🧽 Use eraser gesture to clear parts of the canvas
- ✋ Detect open palm to submit the drawing
- 🤖 Integrates Google Gemini (`gemini-1.5-flash`) for AI-based math solving
- 🖼 Real-time output on webcam with OpenCV

---

## 🛠️ Tech Stack

- **Python**
- **OpenCV**
- **cvzone** (Hand tracking module)
- **Google Generative AI (Gemini)**
- **NumPy**
- **PIL (Python Imaging Library)**

---

## 📦 Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/gesture-math-ai.git
   cd gesture-math-ai
2. **Install dependencies**
    pip install opencv-python cvzone numpy google-generativeai Pillow

3. **Set up Google Generative AI**
    genai.configure(api_key="Your_API")
