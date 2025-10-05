Local CLI Chatbot using Hugging Face

This project implements a **Command-Line Chatbot** powered by a **Hugging Face text-generation model**.  
It maintains **short-term conversational memory** using a **sliding window** to enable coherent multi-turn dialogue.

---

## 🚀 **Overview**

The chatbot loads a small pre-trained model (like `meta-llama/Llama-3.2-1B-Instruct` or `distilgpt2`) and runs locally using the **Transformers** pipeline.  
It remembers the last few user-bot exchanges to provide contextually consistent replies.

---

## 🧩 **Features**

- 🗨️ Real-time interactive chatbot in the terminal  
- 🧠 Sliding-window memory for conversation history  
- ⚡ GPU/CPU auto-detection  
- 🔧 Modular structure for clarity and reusability  
- 🪶 Lightweight model loading via Hugging Face pipeline  
- 🧱 Graceful exit using `/exit` command  

---

## 🏗️ **Project Structure**

```
📁 atg_chatbot/
│
├── model_loader.py     # Loads Hugging Face model and tokenizer
├── chat_memory.py      # Sliding window memory handler
├── interface.py        # Main CLI chatbot interface
├── requirements.txt    # Dependencies list
└── README.md           # Documentation
```

---

## ⚙️ **Setup Instructions**

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/atg-chatbot.git
cd atg-chatbot
```

### 2️⃣ (Optional) Create and activate a virtual environment
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the chatbot
```bash
python interface.py --model meta-llama/Llama-3.2-1B-Instruct
```

---

## 💬 **Example Interaction**

```
User: What is the capital of France?
Bot: The capital of France is Paris.

User: And what about Italy?
Bot: The capital of Italy is Rome.

User: /exit
Exiting chatbot. Goodbye!
```

✅ The chatbot remembers your previous question and continues the conversation naturally.

---

## 🧠 **How It Works**

1. **Model Loading:**  
   Uses `transformers.pipeline()` to load a text-generation model and tokenizer.

2. **Device Detection:**  
   Automatically detects GPU (CUDA) if available, else defaults to CPU.

3. **Memory Management:**  
   Maintains a **deque** (double-ended queue) of the last *n* turns (`max_turns=4` by default).

4. **Prompt Formatting:**  
   Builds contextual prompts like:
   ```
   User: <previous_question>
   Bot: <previous_answer>
   User: <new_question>
   Bot:
   ```

5. **Response Extraction:**  
   Cleans the generated text to return only the bot’s response.

---

## 🧱 **Design Decisions**

- **Sliding-Window Memory:** Keeps recent dialogue context compact and relevant.  
- **Modular Code:** Each script has a clear, isolated purpose.  
- **Lightweight Approach:** Uses existing Hugging Face models; no fine-tuning needed.  
- **Extensible:** Can be easily adapted to GUIs like Gradio or Streamlit.

---

## 🧪 **Testing**

You can quickly test the chatbot after setup:
```bash
python interface.py --model distilgpt2
```
If the model replies and remembers context between turns, your setup is correct.
