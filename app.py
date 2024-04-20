import gradio as gr
from transformers import AutoTokenizer, AutoModel
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

def get_embedding(text):
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        output = model(**tokens)
    embeddings = output.last_hidden_state.mean(1)  # Take mean of embeddings across tokens
    return embeddings

def calculate_similarity(text1, text2):
    embed1 = get_embedding(text1)
    embed2 = get_embedding(text2)
    cos = torch.nn.CosineSimilarity(dim=1)  # Compute similarity across the correct dimension
    similarity = cos(embed1, embed2)
    return f"{similarity.item():.2%} Similarity"

# Create a Gradio interface
iface = gr.Interface(
    fn=calculate_similarity,
    inputs=[gr.inputs.Textbox(label="Input Text 1"), gr.inputs.Textbox(label="Input Text 2")],
    outputs=gr.outputs.Textbox(label="Similarity"),
    title="Text Similarity Checker",
    description="Enter two pieces of text to calculate their semantic similarity."
)

iface.launch()
