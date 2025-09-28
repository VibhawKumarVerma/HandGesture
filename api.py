from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from PIL import Image
import io

app = FastAPI()

# ✅ Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for now allow all, later you can restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
genai.configure(api_key="API_KEY")
model = genai.GenerativeModel('gemini-1.5-flash')

@app.post("/solve")
async def solve_math(file: UploadFile = File(...)):
    image_data = await file.read()
    pil_img = Image.open(io.BytesIO(image_data))
    response = model.generate_content(["Solve this math problem", pil_img])
    return {"answer": response.text}
