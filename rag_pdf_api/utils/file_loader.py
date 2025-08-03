import fitz  # PyMuPDF
import tempfile

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

async def process_pdf(file):
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(await file.read())
    temp.close()

    doc = fitz.open(temp.name)
    all_chunks = []
    for page in doc:
        text = page.get_text()
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
    return all_chunks
