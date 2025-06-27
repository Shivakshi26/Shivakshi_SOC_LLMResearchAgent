import fitz

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    return [{"page": i+1, "text": p.get_text().strip()} for i, p in enumerate(doc)]
