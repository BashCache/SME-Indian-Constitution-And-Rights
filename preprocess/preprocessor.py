# For each of the data in pdf, ppt, text, preprocess them: 
# Remove URLS.
# Remove page numbers.
# Remove headers and footers.
# Remove senences with less than equal to 4 words.
# Save the clean data in a new folder called 'cleaned_data' maintaining the same structure as 'data'.


# Note that for ppts, pdfs, we have already done a majority of preprocessing during the extraction phase.
# This includes removing slide nos, removing dates, removing headr, footer templates, removing reprtitive text etc.
# So here we will just do a final pass of cleaning to remove any remaining URLs, short sentences etc.


import re
from detoxify import Detoxify
import os

toxicity_threshold = 0.5
severity_toxicity_threshold = 0.3
threat_threshold = 0.4
sexual_explicit_threshold = 0.4

def remove_urls(text):
    """Removes URLs (http, https, www) from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_dates(text):
    """Removes dates in various formats (dd-mm-yyyy, dd.mm.yyyy, dd/mm/yyyy, etc.)."""
    # This regex is designed to be broad and catch many common date formats.
    date_pattern = re.compile(
        r'\b('
        # Formats like 12-05-2023, 12/05/2023, 12.05.2023
        r'(\d{1,2}[-./]\d{1,2}[-./]\d{2,4})|'
        # Formats like May 12, 2023 or 12 May 2023
        r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{2,4})|'
        r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4})'
        r')\b',
        re.IGNORECASE
    )
    return date_pattern.sub(r'', text)

def identify_and_remove_toxic_sentence(text, toxicity_threshold=toxicity_threshold, severity_toxicity_threshold=severity_toxicity_threshold, threat_threshold=threat_threshold, sexual_explicit_threshold=sexual_explicit_threshold):
    """Removes sentences with more than a certain number of toxic words."""
    results = Detoxify('multilingual').predict([text])
    print(len(results['toxicity']))
    if results['toxicity'][0] > toxicity_threshold or results['severe_toxicity'][0] > severity_toxicity_threshold or results['threat'][0] > threat_threshold or results['sexual_explicit'][0] > sexual_explicit_threshold:
        print(f"Toxic text: ", {text})
        return " "
    return text

def remove_time(text):
    """Removes time in 24-hour and 12-hour clock formats."""
    # Catches formats like 14:30, 02:30 PM, 2:30pm
    time_pattern = re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM|am|pm)?\b')
    return time_pattern.sub(r'', text)

def clean_punctuation_and_spacing(text):
    """
    Removes excessive punctuation, non-standard chars, extra spaces/tabs,
    and standardizes paragraph breaks (multiple newlines).
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Remove excessive punctuation (keep periods, commas, etc. if needed)
    #    This targets specific chars you listed (!*#) plus maybe others, repeated.
    #    Be careful not to remove single meaningful punctuation.
    text = re.sub(r'[!*#]{2,}', ' ', text) # Remove repetitions of !, *, #
    # Optionally remove isolated symbols if truly noisy:
    # text = re.sub(r'\s[#*]\s', ' ', text)

    # 2. Collapse multiple spaces and tabs into a single space
    #    Use [ \t] to target only spaces and tabs, NOT newlines (\n)
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # 3. Standardize paragraph breaks: Replace 2 or more newlines with exactly two newlines
    text = re.sub(r'\n{3,}', '\n\n', text) # Collapse 3+ newlines to 2

    # 4. Remove spaces/tabs immediately surrounding newlines
    text = re.sub(r'[ \t]+\n', '\n', text) # Remove spaces before newline
    text = re.sub(r'\n[ \t]+', '\n', text) # Remove spaces after newline

    # 5. Remove leading/trailing whitespace (including newlines) from the whole text
    text = text.strip()

    return text

def remove_short_sentences(text, min_words=4):
    """Removes sentences with a word count less than or equal to a threshold."""
    # Split text into sentences based on period, question mark, or exclamation mark
    sentences = re.split(r'(?<=[.?!])\s+', text)

    # Keep sentences that have more than the minimum number of words
    long_sentences = [s for s in sentences if len(s.split()) >= min_words]

    return ' '.join(long_sentences)

def remove_emojis(text: str) -> str:
    """
    Removes emojis and emoticon-related Unicode characters from a string.
    """
    # This comprehensive regex pattern covers most emoji ranges,
    # including symbols, pictographs, transport, and map symbols.
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030"
        "]+",
        flags=re.UNICODE
    )

    # Substitute the matched emoji patterns with an empty string
    return emoji_pattern.sub(r'', text)

# Remove non-English characters apart from punctuations symbols, numbers and spaces.
def remove_non_english_characters(text: str) -> str:
    """Removes non-English characters from the text."""
    return re.sub(r'[^a-zA-Z0-9\s\.,;:!?$#@=|\\\'\"()\[\]{}\-]', '', text)

# Write a method to remove headers and footers if any specific pattern is known.
def remove_headers_and_footers(text: str, header_pattern: str, footer_pattern: str) -> str:
    """Removes headers and footers based on provided patterns."""
    text = re.sub(header_pattern, '', text, flags=re.MULTILINE)
    text = re.sub(footer_pattern, '', text, flags=re.MULTILINE)
    return text

# def identify_and_remove_toxic_sentence(text, max_allowed, toxic_words):
#     """Removes sentences with more than a certain number of toxic words."""
#     pattern = r'\b(' + '|'.join(re.escape(word) for word in toxic_words) + r')\b'
#     sentences = re.split(r'(?<=[۔\.])\s*', text)
#     clean_sentences = [
#         s for s in sentences
#         if s and len(re.findall(pattern, s, re.IGNORECASE)) <= max_allowed
#     ]
#     return " ".join(clean_sentences)

def preprocess_text(text):
    """
    Applies all preprocessing and cleansing steps to the input text.
    """
    if text is None or not isinstance(text, str):
        return ""

    # Apply cleaning functions in a logical order
    toxic_threshold = 2
    text = remove_urls(text)
    text = remove_dates(text)
    text = remove_time(text)
    text = remove_emojis(text)

    # Clean up spacing and punctuation after major removals
    text = clean_punctuation_and_spacing(text)

    # Remove short sentences as the final step
    text = remove_short_sentences(text)

    # text = identify_and_remove_toxic_sentence(text)
    return text

def process_file_in_chunks(input_file: str, output_file: str, batch_size: int = 10):
    """
    Reads a file in batches and applies the master cleaning pipeline to each.
    """
    print(f"\nStarting batch processing for '{input_file}'...")
    batch = []
    batch_count = 0
    total_words = 0
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f_in, \
         open(output_file, 'a', encoding='utf-8') as f_out:

        for line in f_in:
            batch.append(line)
            if len(batch) == batch_size:
                batch_count += 1
                print(f"  -> Cleaning batch #{batch_count}...")
                text_chunk = "".join(batch)
                cleaned_chunk = preprocess_text(text_chunk) # Call the master pipeline
                total_words += len(cleaned_chunk.split(' '))
                print(total_words)
                f_out.write(cleaned_chunk + "\n")
                batch = []

        # Process the final, smaller batch
        if batch:
            batch_count += 1
            print(f"  -> Cleaning final batch #{batch_count}...")
            text_chunk = "".join(batch)
            cleaned_chunk = preprocess_text(text_chunk)
            total_words += len(cleaned_chunk.split(' '))
            f_out.write(cleaned_chunk + "\n")

    print(f"✅ Processing complete. Cleaned data saved to '{output_file}'.")
    return total_words

# For each folder in extracted_data (pdf_extracted, ppt_extracted, text), process all files and save to cleaned_data
def preprocess_all_data(input_base_folder: str, output_base_folder: str):
    """
    Preprocesses all text files in the specified input folder and saves cleaned versions.
    """
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)

    for root, _, files in os.walk(input_base_folder):
        relative_path = os.path.relpath(root, input_base_folder)
        output_folder = os.path.join(output_base_folder, relative_path)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for file in files:
            if file.endswith('.txt'):
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_folder, file)
                print(f"\nProcessing file: {input_file_path}")
                process_file_in_chunks(input_file_path, output_file_path)

if __name__ == "__main__":
    INPUT_BASE_FOLDER = os.path.join("extracted_data\\books_extracted")
    OUTPUT_BASE_FOLDER = os.path.join("cleaned_data\\books_extracted")

    preprocess_all_data(INPUT_BASE_FOLDER, OUTPUT_BASE_FOLDER)