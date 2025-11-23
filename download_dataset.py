import os
from pathlib import Path
from sklearn.datasets import fetch_20newsgroups


def download_dataset():
    print("Downloading 20 Newsgroups dataset...")
    print("This may take a few minutes...")
    
    # Create the data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Downloading the dataset and clean it up
    dataset = fetch_20newsgroups(
        subset='train',
        remove=('headers', 'footers', 'quotes'),  # Remove email headers/footers
        shuffle=False
    )
    
    print(f"\nDataset loaded: {len(dataset.data)} documents")
    print(f"Categories: {len(dataset.target_names)}")
    
    # Saving each document as a separate text file
    print("\nSaving documents to data/ directory...")
    saved_count = 0
    
    for i, (text, target) in enumerate(zip(dataset.data, dataset.target)):
        if not text.strip():  # Skip empty documents
            continue
        
        category = dataset.target_names[target]
        filename = f"doc_{i+1:03d}.txt"
        filepath = data_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        
        saved_count += 1
    
    print(f"\nSaved {saved_count} documents to {data_dir}/")
    print("\nDocuments are named: doc_001.txt, doc_002.txt, ...")
    print("\nYou can now generate embeddings with:")
    print("  python -m src.generate_embeddings")


if __name__ == "__main__":
    download_dataset()

