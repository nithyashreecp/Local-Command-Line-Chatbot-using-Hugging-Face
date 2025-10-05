# interface.py
"""
CLI chatbot that maintains a sliding window memory and uses the text-generation pipeline.
Run: python interface.py
"""

import argparse
from model_loader import load_text_generation_pipeline, detect_device
from chat_memory import ChatMemory
import sys

def build_prompt(context: str, user_input: str) -> str:
    """
    Format the prompt so the model sees the previous exchange and the current user input.
    We'll append 'Bot:' to encourage the model to produce a reply.
    """
    prompt = f"{context}User: {user_input}\nBot:"
    return prompt

def extract_reply(generated_text: str, prompt: str) -> str:
    """
    Since pipeline returns prompt+continuation, strip the prompt out and clean up the reply.
    Stop if model begins 'User:' or 'Bot:' for safety.
    """
    if generated_text.startswith(prompt):
        reply = generated_text[len(prompt):]
    else:
        # fallback
        reply = generated_text
    # trim and avoid returning trailing prompts
    stop_tokens = ["\nUser:", "\nBot:", "User:", "Bot:"]
    for tok in stop_tokens:
        if tok in reply:
            reply = reply.split(tok)[0]
    return reply.strip()

def main():
    parser = argparse.ArgumentParser(description="Local CLI Chatbot using Hugging Face pipeline")
    parser.add_argument("--model", type=str, default="distilgpt2", help="Hugging Face model name (default: distilgpt2)")
    parser.add_argument("--window", type=int, default=4, help="Number of recent turns to remember (default: 4)")
    parser.add_argument("--use-gpu", action="store_true", help="Attempt to use GPU if available")
    parser.add_argument("--max-new-tokens", type=int, default=120, help="Max tokens the model will generate per reply")
    args = parser.parse_args()

    device_id = detect_device(prefer_gpu=args.use_gpu)
    print(f"Loading model '{args.model}' on device {'GPU:'+str(device_id) if device_id >=0 else 'CPU'} ...")
    generator = load_text_generation_pipeline(model_name=args.model, device_id=device_id)
    memory = ChatMemory(max_turns=args.window)

    print("Chatbot ready. Type your message and press Enter. Type /exit to quit.\n")
    try:
        while True:
            user_input = input("User: ").strip()
            if user_input.lower() == "/exit":
                print("Exiting chatbot. Goodbye!")
                break
            if not user_input:
                continue

            context = memory.get_context()
            prompt = build_prompt(context, user_input)

            # generation kwargs - you can tweak these
            gen_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                # ensure models without pad_token_id don't error:
                pad_token_id=getattr(generator.tokenizer, "eos_token_id", None),
            )

            outputs = generator(prompt, **gen_kwargs)
            # pipeline returns a list of results; take the first
            generated_text = outputs[0]["generated_text"]
            reply = extract_reply(generated_text, prompt)
            if not reply:
                reply = "Sorry, I don't have an answer right now."

            print("Bot:", reply)
            memory.add(user_input, reply)

    except KeyboardInterrupt:
        print("\nInterrupted. Exiting chatbot. Goodbye!")
        sys.exit(0)

if __name__ == "__main__":
    main()
