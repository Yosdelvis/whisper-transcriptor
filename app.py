from flask import Flask, request, jsonify
import whisper
import openai
import os

app = Flask(__name__)
model = whisper.load_model("base")
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    file = request.files["audio"]
    file.save("temp.mp3")

    result = model.transcribe("temp.mp3", language="es")
    text = result["text"]

    prompt = f"""
Analiza esta llamada de ventas y señala errores de comunicación, puntos débiles y recomendaciones:
{text}
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    feedback = response["choices"][0]["message"]["content"]
    return jsonify({"transcription": text, "analysis": feedback})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
