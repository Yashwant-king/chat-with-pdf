# ==============================
# Chat with PDF using Hugging Face API
# Free and easy to use
# ==============================

!pip install -q transformers PyPDF2 huggingface_hub torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login
from PyPDF2 import PdfReader
from google.colab import files
import torch

# 1️⃣ Log in to Hugging Face
api_token = input("Enter your Hugging Face API token: ")
login(api_token)

# 2️⃣ Select model (change if you want)
model_name = "tiiuae/falcon-7b-instruct"

# 3️⃣ Load model and tokenizer
print("Loading model (this may take 1-2 minutes)...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 4️⃣ Upload PDF
print("Upload your PDF file:")
uploaded = files.upload()
pdf_path = next(iter(uploaded.keys()))

# 5️⃣ Extract text
reader = PdfReader(pdf_path)
pdf_text = ""
for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        pdf_text += page_text + "\n"

if not pdf_text.strip():
    raise SystemExit("No text extracted. Use a text-based PDF, not scanned images.")

print(f"✅ Extracted {len(pdf_text)} characters from PDF (showing first 500):\n")
print(pdf_text[:500])

# 6️⃣ Ask questions
def ask_question(question):
    prompt = f"""
You are an assistant. Answer using only the following PDF text. If answer not present, say 'I don't know'.

PDF text (partial):
{pdf_text[:15000]}

Question: {question}
"""
    result = generator(prompt, max_length=500, do_sample=True, temperature=0.7)
    return result[0]["generated_text"]

# 7️⃣ Interactive loop
print("\n✅ Ready! Type your questions. Type 'exit' to quit.")
while True:
    q = input("\n❓ Ask: ")
    if q.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break
    ans = ask_question(q)
    print("\n💬 Answer:\n", ans)
