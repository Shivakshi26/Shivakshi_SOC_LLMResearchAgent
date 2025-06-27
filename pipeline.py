from utils.pdf_utils import extract_text_from_pdf
from utils.chunk_utils import chunk_text, embed_chunks, store_in_faiss

def run_pipeline(pdf_path):
    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(pages)
    embeddings, texts = embed_chunks(chunks)
    index = store_in_faiss(embeddings)
    print(f" Stored {len(texts)} chunks in FAISS.")
    return index, texts

if __name__ == "__main__":
    run_pipeline("data/sample_paper.pdf")
