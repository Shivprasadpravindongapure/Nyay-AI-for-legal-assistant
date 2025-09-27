from flask import Flask, request, jsonify, render_template
from llama_cpp import Llama
import os

# ----------------------------
# 1. Initialize Flask App
# ----------------------------
app = Flask(__name__)

# ----------------------------
# 2. Model Loading
# ----------------------------
# Absolute path to your model folder
model_dir = r"C:\Users\prasa\Downloads\Nyay-AI-main\Nyay-AI-main\indian-legal-model"

# Auto-detect the first .gguf file in the folder
gguf_files = [f for f in os.listdir(model_dir) if f.endswith(".gguf")]

if not gguf_files:
    print(f"ERROR: No .gguf model file found in {model_dir}")
    exit()

model_filename = gguf_files[0]  # Picks 'llama-2-7b-chat.Q4_K_M.gguf'
model_path = os.path.join(model_dir, model_filename)

print(f"Loading GGUF model from: {model_path}")

# Load the model
try:
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,  # smaller context for CPU / GPU
        verbose=True,
    )
    print("GGUF Model loaded successfully!")
except Exception as e:
    print(f"ERROR: Failed to load GGUF model.")
    print(f"Python Error: {e}")
    exit()

# ----------------------------
# 3. Flask Routes
# ----------------------------
@app.route("/")
def index():
    # Make sure templates/index.html exists
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    try:
        conversation_history = request.json.get("history", [])

        # Base instruction
        prompt = """### Instruction:
You are Nyay AI, an esteemed legal scholar and expert assistant on Indian Law. You can remember previous parts of the conversation. Your purpose is to provide answers that are well-structured, coherent, and authoritative. Your name is Nyay AI. Maintain a formal and professional tone. Address the user by name if they have provided it in the conversation history.

"""

        # Build conversation history
        for message in conversation_history:
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "user":
                prompt += f"### User:\n{content}\n\n"
            elif role == "assistant":
                prompt += f"### Assistant:\n{content}\n\n"

        # Add final assistant turn
        prompt += "### Assistant:\n"

        # Generate response
        output = llm(
            prompt,
            max_tokens=512,
            echo=False,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["### User:", "### Instruction:"],
        )

        assistant_response = ""
        if "choices" in output and len(output["choices"]) > 0:
            assistant_response = output["choices"][0].get("text", "").strip()

        if not assistant_response:
            assistant_response = "Sorry, I could not generate a response."

        return jsonify({"response": assistant_response})

    except Exception as e:
        print(f"Error during model inference: {e}")
        return jsonify({"response": "Sorry, I encountered an error while generating a response."}), 500

# ----------------------------
# 4. Run Flask App
# ----------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
