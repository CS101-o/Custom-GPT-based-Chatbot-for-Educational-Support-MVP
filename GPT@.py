import fitz  # PyMuPDF
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import tkinter as tk
from tkinter import scrolledtext
from tkinter import filedialog

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Function to read PDF content
def read_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to split context into manageable chunks for the model
def split_into_chunks(context, max_length=512):
    tokens = tokenizer.tokenize(context)
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    return [" ".join(chunk) for chunk in chunks]


def generate_response(question, context=None, max_length=50):
    # Combine the question and context into a single prompt (if context is used)
    prompt_text = f"Question: {question}\nAnswer:"
    if context:
        prompt_text = f"Context: {context[:1000]}\n{prompt_text}"  # Truncate context to fit within model limits

    # Encode the prompt text
    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")

    # Generate a sequence of tokens in response to the prompt
    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=max_length + len(encoded_prompt[0]),
        temperature=1.0,
        top_k=20,
        top_p=0.95,
        repetition_penalty=1.2,
        do_sample=True,
        num_return_sequences=1,
    )

    # Decode the generated sequence to text
    generated_sequence = output_sequences[0].tolist()
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

    # Extract the answer from the generated text
    answer_start = text.find("Answer:") + len("Answer:")
    answer = text[answer_start:].strip()

    return answer


# GUI class for the chatbot
class ChatbotGUI:
    def __init__(self, master):
        self.master = master
        master.title("PDF Chatbot")

        self.conversation_history = scrolledtext.ScrolledText(master, state='disabled', width=70, height=20)
        self.conversation_history.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        self.user_input = tk.Entry(master, width=50)
        self.user_input.grid(row=1, column=0, padx=10, pady=10)

        self.send_button = tk.Button(master, text="Send", command=self.send_message)
        self.send_button.grid(row=1, column=1, padx=10, pady=10)

        self.load_pdf_button = tk.Button(master, text="Load PDF", command=self.load_pdf)
        self.load_pdf_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

        self.pdf_text = ""

    def send_message(self):
        user_text = self.user_input.get()
        if user_text:
            self.update_conversation("You: " + user_text)
            self.user_input.delete(0, tk.END)
            if self.pdf_text:
                bot_response = generate_response(question=user_text, context=self.pdf_text, max_length=200)

                self.update_conversation("Bot: " + bot_response)
            else:
                self.update_conversation("Bot: Please load a PDF file first.")

    def update_conversation(self, message):
        self.conversation_history.config(state='normal')
        self.conversation_history.insert(tk.END, message + "\n")
        self.conversation_history.config(state='disabled')
        self.conversation_history.see(tk.END)

    def load_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if file_path:
            self.pdf_text = read_pdf(file_path)
            self.update_conversation("Bot: PDF loaded successfully.")

# Main function to run the GUI
def main():
    root = tk.Tk()
    gui = ChatbotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
