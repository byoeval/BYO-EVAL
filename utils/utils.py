import sys 
import pandas as pd
from typing import Union
def replace_predictions_with_words(csv_file_path : Union[str, pd.DataFrame], output_file_path : str = None, save : bool =  True):
    """
    Reads a CSV file --> replace the prediction with the word answer

    Args:
        csv_file_path (str): Path to the input CSV file.
        output_file_path (str): Path to save the modified CSV file.
    """
    # a csv path or a csv dataframe
    if isinstance(csv_file_path, str):
        df = pd.read_csv(csv_file_path)
    else:
        df = csv_file_path
    # Check if predictions could be pred, preds, predictions, prediction
    prediction_columns = ['pred', 'preds', 'predictions', 'prediction']
    
    # Find the first matching column name
    pred_col = next((col for col in prediction_columns if col in df.columns), None)
    
    if pred_col:
        for index, row in df.iterrows():
            result = row[pred_col].split()[-1].strip(':')
            if not result:
                print(f"Warning: Empty string found in row {index}")
            df.at[index, pred_col] = result
    else:
        raise ValueError("No prediction column found in DataFrame")
    if save:
        df.to_csv(output_file_path, index=False)
    else:
        return df

# Example usage
if __name__ == "__main__":
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    assert input_csv.endswith('.csv'), "Input file must be a CSV file"
    assert output_csv.endswith('.csv'), "Output file must be a CSV file"
    replace_predictions_with_words(input_csv, output_csv)