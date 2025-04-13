from transformers import AutoTokenizer, pipeline


def chunk_text(text, tokenizer, max_tokens=512):
    tokens = tokenizer.encode(text, truncation=False)
    chunks = [tokens[i : i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]


def summarizer(text, max_length=150, min_length=50):
    summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

    text_chunks = chunk_text(text, tokenizer)
    summaries = []

    print(f"Total Chunks: {len(text_chunks)}")

    for i, chunk in enumerate(text_chunks):
        token_count = len(tokenizer.encode(chunk))
        print(f"Chunk {i + 1}/{len(text_chunks)} - Token Count: {token_count}")

        if token_count < min_length:
            print(f"Skipping chunk {i + 1} due to low token count.")
            continue

        try:
            summary = summarization_pipeline(
                chunk, max_length=max_length, min_length=min_length, do_sample=False
            )
            summaries.append(summary[0]["summary_text"])
        except Exception as e:
            print(f"Error processing chunk {i + 1}: {e}")

    return " ".join(summaries) if summaries else "Summary could not be generated."


def main():
    path = "Superman & Lois (2021) - S01E01 - Pilot.en.srt_cleaned.txt"
    output_path = "summary_output.txt"

    try:
        with open(path, "r", encoding="utf-8") as file:
            text = file.read()

        summary = summarizer(text)
        print("\nFinal Summary:\n", summary)

        with open(output_path, "w", encoding="utf-8") as output_file:
            output_file.write(summary)
        print(f"\nSummary saved to '{output_path}'.")

    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
