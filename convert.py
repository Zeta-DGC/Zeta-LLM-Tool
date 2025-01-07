import json
import re

def extract_sections(content):
    """
    Extracts <Thought> and <Output> sections from a given string.

    Args:
        content (str): The combined content containing <Thought> and <Output>.

    Returns:
        tuple: A tuple containing the thought content and output content.
    """
    # Use regular expressions to extract <Thought> and <Output> sections
    thought_match = re.search(r"<Thought>(.*?)</Thought>", content, re.DOTALL)
    output_match = re.search(r"<Output>(.*?)</Output>", content, re.DOTALL)

    thought_content = thought_match.group(1).strip() if thought_match else ""
    output_content = output_match.group(1).strip() if output_match else ""

    return thought_content, output_content

def convert_jsonl_to_list_format(jsonl_file_path, output_file_path):
    """
    Converts a JSONL file to the specified list-based format.

    Args:
        jsonl_file_path (str): Path to the input JSONL file.
        output_file_path (str): Path to the output JSON file.
    """
    converted_data = []

    with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            # Parse each line as JSON
            record = json.loads(line.strip())

            # Extract instruction and content
            instruction = record["instruction"]
            content = record["output"]

            # Extract <Thought> and <Output> sections from content
            thought_content, output_content = extract_sections(content)

            # Append the reformatted record
            converted_data.append([
                {"role": "input", "content": instruction},
                {"role": "thought", "content": thought_content},
                {"role": "output", "content": output_content}
            ])

    # Save the result to a new JSON file
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(converted_data, output_file, indent=2, ensure_ascii=False)

# Example usage
convert_jsonl_to_list_format("OpenO1-SFT.jsonl", "converted_data.json")
