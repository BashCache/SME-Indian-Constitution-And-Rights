import pandas as pd
import os
import json

def combine_articles(input_csv, output_txt):
    """
    Read CSV file and combine article_id and article_desc in the format:
    Article ID: Article Desc
    """
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Combine the columns in the desired format
    combined_text = []
    for _, row in df.iterrows():
        article_text = f"{row['article_id']}: {row['article_desc']}"
        combined_text.append(article_text)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    
    # Write to output file
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(combined_text))
    
    print(f"Successfully combined {len(combined_text)} articles")
    print(f"Output saved to: {output_txt}")

def process_qa_jsonl(input_jsonl, output_txt):
    """
    Read JSONL file and combine instruction and response in the format:
    instruction: response
    """
    # Read and process the JSONL file
    combined_text = []
    try:
        with open(input_jsonl, 'r', encoding='utf-8', errors='ignore') as f:
            # Try to read as a single JSON array first
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for idx, item in enumerate(data, 1):
                        try:
                            qa_text = f"{item['Instruction']}: {item['Response']}"
                            combined_text.append(qa_text)
                        except KeyError as e:
                            print(f"Warning: Missing key {e} in item {idx}")
                        except Exception as e:
                            print(f"Warning: Error processing item {idx}: {e}")
            except json.JSONDecodeError:
                # If not a JSON array, try reading line by line as JSONL
                f.seek(0)  # Reset file pointer to beginning
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        qa_text = f"{data['Instruction']}: {data['Response']}"
                        combined_text.append(qa_text)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                    except KeyError as e:
                        print(f"Warning: Missing key {e} on line {line_num}")
                    except Exception as e:
                        print(f"Warning: Error processing line {line_num}: {e}")
    except Exception as e:
        print(f"Error reading file {input_jsonl}: {e}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    
    # Write to output file
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(combined_text))

    print(f"Successfully processed {len(combined_text)} QA pairs")
    print(f"Output saved to: {output_txt}")

def process_qa_parquet(input_parquet, output_txt, question_col='question', answer_col='answer'):
    """
    Read Parquet file and combine question and answer columns in the format:
    question: answer
    
    Parameters:
    -----------
    input_parquet : str
        Path to the input parquet file
    output_txt : str
        Path to the output text file
    question_col : str
        Name of the question column in the parquet file
    answer_col : str
        Name of the answer column in the parquet file
    """
    try:
        # Read the parquet file
        df = pd.read_parquet(input_parquet)
        
        # Verify columns exist
        if question_col not in df.columns:
            raise KeyError(f"Question column '{question_col}' not found in parquet file")
        if answer_col not in df.columns:
            raise KeyError(f"Answer column '{answer_col}' not found in parquet file")
        
        # Combine the columns in the desired format
        combined_text = []
        for _, row in df.iterrows():
            qa_text = f"{row[question_col]}: {row[answer_col]}"
            combined_text.append(qa_text)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_txt), exist_ok=True)
        
        # Write to output file
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(combined_text))
        
        print(f"Successfully processed {len(combined_text)} QA pairs from parquet")
        print(f"Output saved to: {output_txt}")
        
    except Exception as e:
        print(f"Error processing parquet file {input_parquet}: {e}")

if __name__ == "__main__":
    # Set up paths for articles
    input_csv = os.path.join("data", "text_versions", "Articles_Indian_Constitution.csv")
    output_articles = os.path.join("extracted_data", "text_extracted", "Articles_Indian_Constitution.txt")
    
    # Set up paths for QA pairs from JSONL
    input_jsonl = os.path.join("data", "text_versions", "Indian_Law_Basic_QA.jsonl")
    output_qa_jsonl = os.path.join("extracted_data", "text_extracted", "Indian_Law_Basic_QA.txt")
    
    # Set up paths for QA pairs from Parquet
    input_parquet = os.path.join("data", "text_versions", "Indian_Constitution_QA_Gemini.parquet")
    output_qa_parquet = os.path.join("extracted_data", "text_extracted", "Indian_Constitution_QA_Gemini.txt")
    
    # Process all files
    print("Processing Articles...")
    # combine_articles(input_csv, output_articles)
    
    print("\nProcessing QA pairs from JSONL...")
    # process_qa_jsonl(input_jsonl, output_qa_jsonl)
    
    print("\nProcessing QA pairs from Parquet...")
    process_qa_parquet(input_parquet, output_qa_parquet)