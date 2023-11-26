from openai import OpenAI
import pandas as pd
import tiktoken
import numpy as np
from ast import literal_eval
import PyPDF2
import csv




# see: https://platform.openai.com/docs/guides/embeddings/use-cases
# see: https://cookbook.openai.com/examples/get_embeddings_from_dataset
# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
client = OpenAI()
input_datapath = "documents/chunked_text.csv"

def main():
    # chunk_pdf("documents/combined.pdf")
    # create_embeddings()
    question = "How can I get a longer term visa?"
    relevant_chunks = search_legal(question, pprint=False)
    print(ask_question(question, relevant_chunks))


def ask_question(question, relevant_chunks):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a legal assistant, skilled in explaining complex legal issues regarding employment and immigration in South Korea in a simple and understandable way. Do not mention that context was given."},
                {"role": "user", "content": "Using the context given below, answer this question: " + question + "\n" + "\n".join(relevant_chunks)}
            ]
        )      

    return completion.choices[0].message


def chunk_pdf(pdf, limit = 500, overlap = 100):
    chunks = []
    chunk = ""
    with open(pdf, "rb") as f:
        pdf = PyPDF2.PdfReader(f)
        for page in pdf.pages:
            chunk += page.extract_text()
            while (len(chunk)) > limit:
                chunks.append(chunk[:limit])
                chunk = chunk[limit-overlap]
        if len(chunk):
            chunks.append(chunk)
    return chunks

def save_chunks(chunks):
    file = open("documents/chunked_text.csv", 'w+', newline="")
    with file:
        write = csv.writer(file)
        write.writerow(["text"])
        for chunk in chunks:
            write.writerow([chunk])



# See https://cookbook.openai.com/examples/semantic_text_search_using_embeddings
def search_legal(question, n=15, pprint=True):
    embeddings_path = "documents/combined_with_embeddings.csv"
    df = pd.read_csv(embeddings_path)
    df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)
    product_embedding = get_embedding(
        question,
        model="text-embedding-ada-002"
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .text
    )

    relevant_chunks = []
    for r in results:
        relevant_chunks.append(r)
    
    if pprint:
        for r in results:
            print(r[:200])
            print()

    return relevant_chunks

# https://github.com/openai/openai-cookbook/blob/main/examples/utils/embeddings_utils.py
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def create_embeddings():
    df = pd.read_csv(input_datapath)
    df = df.dropna()
    df = df[["text"]]
    df.head(1)
    encoding = tiktoken.get_encoding(embedding_encoding)
    # Omit anything lesss than 40 characters (10 tokens)
    df["n_tokens"] = df.text.apply(lambda x: len(encoding.encode(x)))
    df = df[df.n_tokens >= 10]
    df["embedding"] = df.text.apply(lambda x: get_embedding(x, model=embedding_model))
    df.to_csv("documents/combined_with_embeddings.csv")
    print(len(df))

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding


if __name__ == "__main__":
    main()