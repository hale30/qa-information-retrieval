import os
import re


def process_markdown_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    question_count = 0
    updated_lines = []

    for line in lines:
        # Normalize excessive tabs to single space
        line = re.sub(r'^\t+', '', line)
        updated_lines.append(line)

        # Add numbering to "### Question" lines
        # if line.strip() == "### Question":
        #     question_count += 1
        #     line = f"### Question {question_count}\n"

        # updated_lines.append(line)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)


def process_all_markdown_files():
    for file_name in os.listdir(''):
        if file_name.endswith('.md'):
            process_markdown_file(file_name)
            print(f"Processed: {file_name}")


if __name__ == "__main__":
    process_all_markdown_files()
