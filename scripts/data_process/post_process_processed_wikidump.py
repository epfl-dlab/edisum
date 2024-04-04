import re

def remove_pattern_from_file(file_path):
    # Read the file content
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Define the pattern to match
    pattern = r"/\*.*?\*/"
    
    # Remove the matched pattern from the content
    modified_content = re.sub(pattern, "", content)
    
    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.write(modified_content)
        
def filter_empty_edits(file_path):
    import pandas as pd

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Remove rows with empty edit_summary
    df = df[df['edit_summary'].str.strip().astype(bool)]

    # Save the filtered data to a new CSV file
    df.to_csv('../data/wikidump_processed/2022_filter_plaintext_all_final_final.csv', index=False)

# Example usage
file_path = '../data/wikidump_processed/2022_filter_plaintext_all_final.csv'

# Remove the pattern from the file
remove_pattern_from_file(file_path)
filter_empty_edits(file_path)
