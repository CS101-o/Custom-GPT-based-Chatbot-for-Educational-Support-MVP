import fitz  # PyMuPDF
import torch
from transformers import RobertaTokenizer, RobertaForQuestionAnswering
import tkinter as tk
from tkinter import scrolledtext
from tkinter import filedialog

# Initialize tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('deepset/roberta-base-squad2')
model = RobertaForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2')

# Function to read PDF content
def read_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text



def split_into_chunks(context, max_length=512, stride=256):
    """
    Splits the context into chunks, with a specified overlap (stride) between them.
    """
    tokens = tokenizer.tokenize(context)
    chunk_size = max_length - stride
    chunks = []

    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i:i + max_length]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))

    return chunks

def answer_question(question, context, max_length=512, stride=256):
    """
    Processes the context in chunks to find the best answer, considering an overlap (stride) between chunks.
    """
    chunked_contexts = split_into_chunks(context, max_length, stride)
    best_answer = ""
    highest_score = float('-inf')

    for chunk in chunked_contexts:
        # Encode the question and chunk together
        inputs = tokenizer.encode_plus(question, chunk, add_special_tokens=True, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].tolist()[0]

        # Get model outputs
        outputs = model(**inputs)
        answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits

        # Determine the start and end positions of the answer in the chunk
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        answer_score = answer_start_scores[:, answer_start] + answer_end_scores[:, answer_end - 1]

        # Update the best answer based on score
        if answer_score > highest_score:
            highest_score = answer_score.item()
            answer_tokens = input_ids[answer_start:answer_end]
            best_answer = tokenizer.decode(answer_tokens)

    return best_answer if best_answer else "Sorry, I couldn't find an answer in the document."


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
                bot_response = answer_question(user_text, self.pdf_text)
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
