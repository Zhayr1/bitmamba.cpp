import struct
from transformers import GPT2Tokenizer

def export_vocab_to_bin(output_path="tokenizer.bin"):
    print("Loading tokenizer from HuggingFace...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    vocab_size = 50257
    
    print(f"Exporting {vocab_size} tokens to '{output_path}'...")
    
    with open(output_path, "wb") as f:
        for i in range(vocab_size):
            token_text = tokenizer.decode([i], clean_up_tokenization_spaces=False)
            token_bytes = token_text.encode('utf-8')
            length = len(token_bytes)
            f.write(struct.pack('<I', length))
            f.write(token_bytes)

    print("Done! Tokenizer saved to 'tokenizer.bin'.")

if __name__ == "__main__":
    export_vocab_to_bin()
