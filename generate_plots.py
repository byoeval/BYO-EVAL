import os
import pandas as pd
import re
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from evaluation_pipeline.metrics_plotter import MetricsPlotter

def clean_prediction_data(df, dataframe_type):
    """
    Clean prediction data based on dataframe type.
    
    Args:
        df: DataFrame with 'pred' and 'target' columns
        dataframe_type: Type of dataframe ('identification' or 'counting')
    
    Returns:
        Cleaned DataFrame
    """
    df_cleaned = df.copy()
    
    # Fix column names if needed - some files may have 'prediction' instead of 'pred'
    # or other variations
    if 'prediction' in df_cleaned.columns and 'pred' not in df_cleaned.columns:
        df_cleaned['pred'] = df_cleaned['prediction']
    if 'answer' in df_cleaned.columns and 'pred' not in df_cleaned.columns:
        df_cleaned['pred'] = df_cleaned['answer']
    if 'ground_truth' in df_cleaned.columns and 'target' not in df_cleaned.columns:
        df_cleaned['target'] = df_cleaned['ground_truth']
    
    # Remove curly braces and other special characters that appear in some predictions
    if 'pred' in df_cleaned.columns:
        df_cleaned['pred'] = df_cleaned['pred'].astype(str).str.replace(r'[{}]', '', regex=True).str.strip()
    
    if 'identification' in dataframe_type:
        # For identification, both should be chess piece names
        chess_pieces = ['pawn', 'rook', 'knight', 'bishop', 'queen', 'king']
        
        # Convert predictions to lowercase and remove special characters
        if 'pred' in df_cleaned.columns:
            df_cleaned['pred'] = df_cleaned['pred'].astype(str).str.lower()
            df_cleaned['pred'] = df_cleaned['pred'].apply(lambda x: next((piece for piece in chess_pieces if piece in x), x))
        
        # Convert targets to lowercase if needed
        if 'target' in df_cleaned.columns:
            df_cleaned['target'] = df_cleaned['target'].astype(str).str.lower()
    
    else:  # Counting or localization
        # Target should already be an int, but let's ensure
        print("CLEANING FOR COUNT OR LOCALIZATION")
        print(df_cleaned.columns)
        if 'target' in df_cleaned.columns:
            df_cleaned['target'] = pd.to_numeric(df_cleaned['target'], errors='coerce')
        print("TARGET IS NUMERIC")
        print(df_cleaned['target'].head(3))
        # Convert string predictions to integers
        if 'pred' in df_cleaned.columns:
            print("PRED IS IN COLUMNS")
            # First try direct conversion
            temp_preds = pd.to_numeric(df_cleaned['pred'], errors='coerce')
            print("TEMP PREDS")
            print(temp_preds.head(3))
            # For values that couldn't be converted directly
            for idx in df_cleaned.index[temp_preds.isna()]:
                original_value = str(df_cleaned.at[idx, 'pred']).lower()
                
                # Handle word numbers
                number_words = {
                    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 
                    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
                    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
                    'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20
                }
                
                for word, num in number_words.items():
                    if word in original_value:
                        temp_preds[idx] = num
                        break
                
                # Extract number using regex for cases like "6}"
                if pd.isna(temp_preds[idx]):
                    match = re.search(r'\d+', original_value)
                    if match:
                        temp_preds[idx] = int(match.group())
            
            df_cleaned['pred'] = temp_preds
            print("DF CLEANED")
            print(df_cleaned['pred'].head(3))
    print("RETURNING DF CLEANED")
    print(df_cleaned.head(3))
    return df_cleaned

# Get all CSV files from diagnostic_results
diagnostic_dir = "diagnostic_results"
csv_files = []

# also add all csv files of all subdirectories
for root, dirs, files in os.walk(diagnostic_dir):
    for file in files:
        if file.endswith('.csv'):
            csv_files.append(os.path.join(root, file))



# remove csv files with "call_metrics " in the filename
csv_files = [file for file in csv_files if "call_metrics" not in file]
csv_files = [file for file in csv_files if "mini" not in file]
csv_files = [file for file in csv_files if "count" in file]
csv_files = [file for file in csv_files if "results" in file]

# remove "diagnostic_results" from the csv files
output_names = [file.replace("diagnostic_results/", "") for file in csv_files]
# remove / and replace with __
output_names = [name.replace("/", "__") for name in output_names]


print(f"Found {len(csv_files)} CSV files to process")
print(csv_files)

df_type = "count"


# Loop over all CSV files
for i, csv_file in enumerate(csv_files):
    print(f"\nProcessing {csv_file}")

    # Create output directory following the pattern plots/name_of_results_dir
    # For files in the root directory, use the filename
    # For files in subdirectories, use the parent directory name
    dir_path = os.path.dirname(csv_file)
    parent_dir = os.path.basename(dir_path)

    output_dir = os.path.join("plots_test", output_names[i])
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load dataframe
    df = pd.read_csv(csv_file)
    print(f"Loaded dataframe with {len(df)} rows")

    # Print column names for debugging
    print(f"Columns in CSV: {df.columns.tolist()}")

    # Clean data
    df_cleaned = clean_prediction_data(df, df_type)
    print(f"Cleaned dataframe. Sample pred/target pairs:")
    for j in range(min(3, len(df_cleaned))):
        if j < len(df):
            orig_pred = df.iloc[j]['pred'] if 'pred' in df.columns else 'N/A'
            orig_target = df.iloc[j]['target'] if 'target' in df.columns else 'N/A'
            cleaned_pred = df_cleaned.iloc[j]['pred'] if 'pred' in df_cleaned.columns else 'N/A'
            cleaned_target = df_cleaned.iloc[j]['target'] if 'target' in df_cleaned.columns else 'N/A'
            print(f"  Original: {orig_pred} → {orig_target}, Cleaned: {cleaned_pred} → {cleaned_target}")

    # Ensure required columns are present
    if 'pred' not in df_cleaned.columns or 'target' not in df_cleaned.columns:
        raise ValueError(f"Required columns 'pred' and/or 'target' not found in {csv_file}")

    # Print head of df only for columns pred and target
    print(df_cleaned.columns)
    print(df_cleaned[["image",'pred', 'target']].head(3))

    # Duplicate variable target as "number"
    #df_cleaned['number'] = df_cleaned['target']

    # Add a note about the original file in the output directory
    with open(os.path.join(output_dir, "source_info.txt"), "w") as f:
        f.write(f"Source file: {csv_file}\n")
        f.write(f"Data type: {df_type}\n")
        f.write(f"Row count: {len(df)}\n")
        f.write(f"Columns: {', '.join(df.columns.tolist())}\n")

    plot_choices = {
            'mae': True,
            'accuracy': True,
            'pred_vs_target': True,
            'error_distribution': True,
            'confusion_matrix': True,
            'nmae': True,
        }
    
    # Choose variable to plot based on dataframe type
    if df_type == "count":
        df_cleaned['number of objects'] = df_cleaned['target']
        variables_to_plot = ['number of objects']
        if "blur" in csv_file:
            variables_to_plot.append("blur")
        if "vertical" in csv_file:
            print("VERTICAL")
            variables_to_plot.append("vertical_overlap")
        if "horizontal" in csv_file:
            print("HORIZONTAL")
            variables_to_plot.append("horizontal_overlap")
    
    elif df_type == "identification":
        plot_choices = {
            'mae': False,
            'nmae': False,
            'accuracy': True,
            'pred_vs_target': True,
            'error_distribution': True,
            'confusion_matrix': True,
        }
        variables_to_plot = ['target']
    elif df_type == "localization":
        df_cleaned["distance"] = df_cleaned["target"]
    # Initialize the MetricsPlotter with chosen variable
    variable_col = variables_to_plot[0]

    # Make sure pred and target columns are prepared for plotting
    if pd.api.types.is_numeric_dtype(df_cleaned[variable_col]):
        print(f"Variable '{variable_col}' is numeric with values: {sorted(df_cleaned[variable_col].unique())[:10]}")
    else:
        print(f"Variable '{variable_col}' is non-numeric with values: {sorted(df_cleaned[variable_col].unique())[:10]}")

    # Initialize plotter
    plotter = MetricsPlotter(df=df_cleaned, variable_col=variable_col, output_dir=output_dir)

    # Create advanced plotting options if we have multiple variables
    advanced_plot_options = None
    if len(variables_to_plot) > 1:
        print(f"Detected {len(variables_to_plot)} variables, setting up cross-variable and conditional plots")
        
        # Convert plot_choices to metrics list for cross-variable plots
        cross_variable_metrics = []
        if plot_choices.get('mae', False):
            cross_variable_metrics.append('mae')
        if plot_choices.get('nmae', False):
            cross_variable_metrics.append('nmae')
        if plot_choices.get('accuracy', False):
            cross_variable_metrics.append('mean_accuracy')
        # Always include count for context
        if 'count' not in cross_variable_metrics:
            cross_variable_metrics.append('count')
            
        # Convert plot_choices to metrics list for conditional plots
        conditional_metrics = []
        if plot_choices.get('mae', False):
            conditional_metrics.append('mae')
        if plot_choices.get('nmae', False):
            conditional_metrics.append('nmae')
        if plot_choices.get('accuracy', False):
            conditional_metrics.append('accuracy')
            
        # If no metrics selected, use defaults
        if not cross_variable_metrics:
            cross_variable_metrics = ['mae', 'mean_accuracy', 'count']
        if not conditional_metrics:
            conditional_metrics = ['mae', 'accuracy']
            
        # Create all pairs of variables for cross-variable analysis
        cross_variable_configs = []
        conditional_configs = []
        
        for i in range(len(variables_to_plot)):
            for j in range(i+1, len(variables_to_plot)):
                var1 = variables_to_plot[i]
                var2 = variables_to_plot[j]
                
                # Add cross-variable plot for this pair
                cross_variable_configs.append({
                    "variable1": var1,
                    "variable2": var2,
                    "metrics": cross_variable_metrics,
                    "output_dir": os.path.join(output_dir, f"cross_{var1}_vs_{var2}"),
                    "save_individual_pngs": True,
                    "save_combined_pdf": True
                })
                
                # Add conditional plots in both directions
                conditional_configs.append({
                    "primary_variable": var1,
                    "conditional_variable": var2,
                    "output_dir": os.path.join(output_dir, f"conditional_{var1}_by_{var2}"),
                    "metrics": conditional_metrics,
                    "save_individual_pngs": True
                })
                
                conditional_configs.append({
                    "primary_variable": var2,
                    "conditional_variable": var1,
                    "output_dir": os.path.join(output_dir, f"conditional_{var2}_by_{var1}"),
                    "metrics": conditional_metrics,
                    "save_individual_pngs": True
                })
                
        # Combine into the advanced_plot_options dictionary
        advanced_plot_options = {
            "cross_variable_plots": cross_variable_configs,
            "conditional_plots": conditional_configs
        }

    # Run the plotting suite
    print(f"Running plotting suite with variables: {variables_to_plot}")
    plotter.run_plotting_suite(
        variables_to_plot_individually=variables_to_plot,
        plot_choices=plot_choices,
        suite_output_dir=output_dir,
        save_combined_pdf_per_variable=True,
        save_individual_pngs_per_variable=True,
        advanced_plot_options=advanced_plot_options
    )
    print(f"Completed plotting for {csv_file}")


print("\nAll processing complete!")
