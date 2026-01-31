import argparse
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

from transformers import GPT2Tokenizer

def main():
    parser = argparse.ArgumentParser(description="Decode a list of token IDs into text using GPT-2 tokenizer.")
    parser.add_argument("tokens", type=str, help="List of tokens separated by spaces (e.g., '15496 11 314')")
    args = parser.parse_args()

    # 1. Load GPT-2 Tokenizer (will download vocab.json if first time)
    print("Loading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # 2. Process input string
    raw_input = args.tokens
    
    try:
        # Convert string "15496 11..." to list of integers [15496, 11, ...]
        token_ids = [int(t) for t in raw_input.split()]
    except ValueError:
        print("Error: Input must contain only numbers separated by spaces.")
        return

    # 3. Decode
    decoded_text = tokenizer.decode(token_ids)

    # 4. Show result
    print("\n" + "="*30)
    print(f"Input IDs: {token_ids}")
    print("-" * 30)
    print(f"Decoded Text: \n\n{decoded_text}")
    print("\n" + "="*30)

if __name__ == "__main__":
    main()
