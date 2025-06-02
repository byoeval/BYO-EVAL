from typing import Any

import numpy as np
import pandas as pd


def compute_regression_stats(preds: np.ndarray | pd.Series, targets: np.ndarray | pd.Series) -> dict[str, Any]:
    """
    Compute extensive regression and distribution statistics for predictions and targets.
    Handles missing values robustly.
    If either preds or targets has zero standard deviation, correlation is set to np.nan to avoid division by zero warnings.
    """
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    mask = ~np.isnan(preds) & ~np.isnan(targets)
    preds = preds[mask]
    targets = targets[mask]
    n = len(preds)
    if n == 0:
        return {k: np.nan for k in [
            'mae', 'mae_std', 'mse', 'rmse', 'nmae', 'mean_error', 'std_error', 'median_error', 'max_error', 'min_error',
            'mean_abs_error', 'median_abs_error', 'max_abs_error', 'min_abs_error',
            'pred_mean', 'pred_std', 'pred_median', 'pred_min', 'pred_max',
            'target_mean', 'target_std', 'target_median', 'target_min', 'target_max',
            'corr_pred_target', 'r2_score', 'skew_error', 'kurtosis_error',
            'outlier_count', 'p10_error', 'p25_error', 'p75_error', 'p90_error',
            'mean_accuracy', 'std_accuracy', 'count']}
    errors = preds - targets
    abs_errors = np.abs(errors)
    mae = np.mean(abs_errors)
    mae_std = np.std(abs_errors)
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    nmae = mae / (targets.max() - targets.min()) if targets.max() != targets.min() else (mae / np.abs(targets.mean()) if targets.mean() != 0 else np.nan)
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    median_error = np.median(errors)
    max_error = np.max(errors)
    min_error = np.min(errors)
    mean_abs_error = np.mean(abs_errors)
    median_abs_error = np.median(abs_errors)
    max_abs_error = np.max(abs_errors)
    min_abs_error = np.min(abs_errors)
    pred_mean = np.mean(preds)
    pred_std = np.std(preds)
    pred_median = np.median(preds)
    pred_min = np.min(preds)
    pred_max = np.max(preds)
    target_mean = np.mean(targets)
    target_std = np.std(targets)
    target_median = np.median(targets)
    target_min = np.min(targets)
    target_max = np.max(targets)

    outlier_count = int(np.sum(abs_errors > (2 * np.std(abs_errors)))) if n > 1 else 0
    percentiles = np.percentile(errors, [10, 25, 75, 90]) if n > 0 else [np.nan]*4
    # Accuracy metrics (for regression, define as exact match or within tolerance)
    tolerance = 1e-6
    accuracy = np.abs(errors) < tolerance
    mean_accuracy = np.mean(accuracy)
    std_accuracy = np.std(accuracy)
    return {
        'mae': mae,
        'mae_std': mae_std,
        'mse': mse,
        'rmse': rmse,
        'nmae': nmae,
        'mean_error': mean_error,
        'std_error': std_error,
        'median_error': median_error,
        'max_error': max_error,
        'min_error': min_error,
        'mean_abs_error': mean_abs_error,
        'median_abs_error': median_abs_error,
        'max_abs_error': max_abs_error,
        'min_abs_error': min_abs_error,
        'pred_mean': pred_mean,
        'pred_std': pred_std,
        'pred_median': pred_median,
        'pred_min': pred_min,
        'pred_max': pred_max,
        'target_mean': target_mean,
        'target_std': target_std,
        'target_median': target_median,
        'target_min': target_min,
        'target_max': target_max,
        'outlier_count': outlier_count,
        'p10_error': percentiles[0],
        'p25_error': percentiles[1],
        'p75_error': percentiles[2],
        'p90_error': percentiles[3],
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'count': n
    }

class MetricsCalculator:
    """
    Calculates various metrics based on prediction and target values in a DataFrame.
    """

    def __init__(self, df: pd.DataFrame, group_by_cols: list[str] | None = None):
        """
        Initializes the MetricsCalculator with a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing 'pred' and 'target' columns,
                               and optionally columns specified in group_by_cols.
            group_by_cols (Optional[List[str]]): List of column names to group by for detailed metrics.
                                                 These columns should exist in df.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        # check if columns are pred and target and add an s if not
        if 'pred' in df.columns and 'target' in df.columns:
            df.rename(columns={'pred': 'preds', 'target': 'targets'}, inplace=True)
        elif 'pred' in df.columns:
            df.rename(columns={'pred': 'preds'}, inplace=True)
        elif 'target' in df.columns:
            df.rename(columns={'target': 'targets'}, inplace=True)

        if 'preds' not in df.columns or 'targets' not in df.columns:
            raise ValueError("DataFrame must contain 'preds' and 'targets' columns.")

        self.df = df.copy()
        self.target_type: str = "unknown"
        self.target_dims: int = 0
        self.is_numeric_tuple: bool = False
        self._analyze_target_type() # Analyzes based on the whole df passed to this instance

        self.group_by_cols = group_by_cols
        if self.group_by_cols:
            for col in self.group_by_cols:
                if col not in self.df.columns:
                    raise ValueError(f"Group-by column '{col}' not found in DataFrame columns: {self.df.columns.tolist()}")

    def _analyze_target_type(self) -> None:
        """
        Analyzes the 'targets' column to determine its type and dimensionality.
        Sets self.target_type, self.target_dims, and self.is_numeric_tuple.
        """
        if self.df['targets'].empty:
            self.target_type = "empty"
            self.target_dims = 0
            return

        first_valid_target = None
        for target in self.df['targets']:
            if target is not None: # Handles potential None values if any
                 # Check for pd.NA separately as it's not None but behaves like NaN
                if pd.isna(target):
                    continue
                first_valid_target = target
                break

        if first_valid_target is None: # All targets are None or pd.NA
            self.target_type = "all_na"
            self.target_dims = 0
            return

        if isinstance(first_valid_target, int | float):
            self.target_type = "numeric"
            self.target_dims = 1
        elif isinstance(first_valid_target, str):
            self.target_type = "string"
            self.target_dims = 1
        elif isinstance(first_valid_target, tuple):
            self.target_type = "tuple"
            self.target_dims = len(first_valid_target)
            if self.target_dims > 0:
                self.is_numeric_tuple = all(isinstance(x, int | float) for x in first_valid_target)
            else: # Empty tuple
                self.is_numeric_tuple = False
        else:
            self.target_type = "other"
            self.target_dims = 1 # Or handle as an error/unknown

    def _calculate_numeric_metrics_for_series(self, targets: pd.Series, preds: pd.Series) -> dict[str, float]:
        """
        Calculates numeric metrics for a single dimension (series).

        Args:
            targets (pd.Series): Series of true target values.
            preds (pd.Series): Series of predicted values.

        Returns:
            Dict[str, float]: Dictionary with MAE, NMAE, MSE, RMSE.
        """
        if not pd.api.types.is_numeric_dtype(targets) or not pd.api.types.is_numeric_dtype(preds):
             # Attempt conversion, coerce errors to NaN
            targets = pd.to_numeric(targets, errors='coerce')
            preds = pd.to_numeric(preds, errors='coerce')

            # Drop rows where conversion failed in either series to ensure alignment
            valid_indices = targets.notna() & preds.notna()
            targets = targets[valid_indices]
            preds = preds[valid_indices]

            if targets.empty or preds.empty:
                return {
                    "mae": np.nan, "nmae": np.nan,
                    "mse": np.nan, "rmse": np.nan,
                    "count": 0
                }

        errors = targets - preds
        mae = np.mean(np.abs(errors))
        mse = np.mean(errors**2)
        rmse = np.sqrt(mse)

        target_range = targets.max() - targets.min()
        if target_range > 0:
            nmae = mae / target_range
        elif targets.mean() != 0 :
            nmae = mae / abs(targets.mean()) if targets.mean() != 0 else (0.0 if mae == 0 else np.nan)
        else: # Range is 0 and mean is 0
            nmae = 0.0 if mae == 0 else np.nan # If MAE is also 0, then NMAE is 0, else undefined

        return {
            "mae": float(mae), "nmae": float(nmae),
            "mse": float(mse), "rmse": float(rmse),
            "count": len(targets)
        }

    def calculate_numeric_metrics(self) -> dict[str, float] | list[dict[str, float]] | None:
        """
        Calculates numeric metrics (MAE, NMAE, MSE, RMSE).

        Returns:
            Union[Dict[str, float], List[Dict[str, float]], None]:
            Metrics for single numeric targets, list of metrics for tuple targets (per dimension),
            or None if targets are not numeric.
        """
        if self.target_type == "numeric":
            # Ensure 'targets' and 'preds' are numeric, coercing errors to NaN
            targets_numeric = pd.to_numeric(self.df['targets'], errors='coerce')
            preds_numeric = pd.to_numeric(self.df['preds'], errors='coerce')

            # Filter out NaNs that may have resulted from coercion or were pre-existing
            valid_idx = targets_numeric.notna() & preds_numeric.notna()
            targets_filtered = targets_numeric[valid_idx]
            preds_filtered = preds_numeric[valid_idx]

            if targets_filtered.empty:
                return {
                    "mae": np.nan, "nmae": np.nan,
                    "mse": np.nan, "rmse": np.nan,
                    "count": 0
                }
            return self._calculate_numeric_metrics_for_series(targets_filtered, preds_filtered)

        elif self.target_type == "tuple" and self.is_numeric_tuple and self.target_dims > 0:
            all_dim_metrics = []
            for i in range(self.target_dims):
                try:
                    targets_dim_i = self.df['targets'].apply(lambda x: x[i] if isinstance(x, tuple) and len(x) > i else np.nan)
                    preds_dim_i = self.df['preds'].apply(lambda x: x[i] if isinstance(x, tuple) and len(x) > i else np.nan)
                except IndexError: # Should not happen if _analyze_target_type is correct
                    # This handles cases where some tuples might be shorter than expected, though
                    # _analyze_target_type assumes consistent tuple length based on the first valid one.
                    # For robustness, we catch this and effectively skip the dimension or mark as NaN.
                    all_dim_metrics.append({
                        "mae": np.nan, "nmae": np.nan,
                        "mse": np.nan, "rmse": np.nan,
                        "count": 0, "dimension_error": "IndexError or inconsistent tuple length"
                    })
                    continue

                # Coerce to numeric and handle NaNs
                targets_dim_i_numeric = pd.to_numeric(targets_dim_i, errors='coerce')
                preds_dim_i_numeric = pd.to_numeric(preds_dim_i, errors='coerce')

                valid_idx = targets_dim_i_numeric.notna() & preds_dim_i_numeric.notna()
                targets_filtered = targets_dim_i_numeric[valid_idx]
                preds_filtered = preds_dim_i_numeric[valid_idx]

                if targets_filtered.empty:
                    all_dim_metrics.append({
                        "mae": np.nan, "nmae": np.nan,
                        "mse": np.nan, "rmse": np.nan,
                        "count": 0
                    })
                else:
                    all_dim_metrics.append(self._calculate_numeric_metrics_for_series(targets_filtered, preds_filtered))
            return all_dim_metrics
        else:
            return None # Not numeric or numeric tuple

    def calculate_accuracy_metrics(self) -> dict[str, float]:
        """
        Calculates accuracy metrics.

        Returns:
            Dict[str, float]: Dictionary with mean accuracy and standard deviation of the accuracy indicator.
        """
        if self.df.empty:
            return {"mean_accuracy": np.nan, "std_accuracy": np.nan, "count": 0}

        # Handle potential pd.NA in direct comparison.
        # A robust way is to apply a function row-wise.
        def safe_compare(row):
            pred, target = row['preds'], row['targets']
            if pd.isna(pred) and pd.isna(target):
                return True # Or False, depending on desired NA handling, let's say NA != NA
            if pd.isna(pred) or pd.isna(target):
                return False
            try:
                return pred == target
            except TypeError: # For complex types that might not support == directly or raise errors
                return False


        is_correct = self.df.apply(safe_compare, axis=1)

        count = len(is_correct)
        if count == 0:
            return {"mean_accuracy": np.nan, "std_accuracy": np.nan, "count": 0}

        mean_accuracy = np.mean(is_correct)
        std_accuracy = np.std(is_correct.astype(float)) # std of 0/1 indicators

        return {
            "mean_accuracy": float(mean_accuracy),
            "std_accuracy": float(std_accuracy),
            "count": count
        }

    def calculate_all_metrics(self) -> dict[str, Any]:
        """
        Calculates all relevant metrics based on the target type.

        Returns:
            Dict[str, Any]: A dictionary containing all computed metrics.
        """
        results = {
            "target_type_info": {
                "type": self.target_type,
                "dimensions": self.target_dims,
                "is_numeric_tuple": self.is_numeric_tuple
            },
            "accuracy_metrics": self.calculate_accuracy_metrics()
        }

        numeric_metrics_result = self.calculate_numeric_metrics()
        if numeric_metrics_result is not None:
            results["numeric_metrics"] = numeric_metrics_result

        return results

    def calculate_grouped_metrics(self) -> list[dict[str, Any]]:
        """
        Calculates metrics for each group defined by `group_by_cols`.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary contains
                                  'group_values' (a dict of column:value for the group) and
                                  'metrics' (the result of calculate_all_metrics() for that group).
                                  Returns an empty list if group_by_cols was not provided or is empty.
        """
        if not self.group_by_cols:
            # Or print a warning, or return None, depending on desired behavior
            return []

        # Ensure all group_by_cols actually exist to prevent KeyError in groupby
        missing_cols = [col for col in self.group_by_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"One or more group_by columns not found in DataFrame: {missing_cols}")

        try:
            # dropna=False ensures that groups with NaN keys are also included if they exist.
            # If NaNs in grouping keys should exclude rows, set dropna=True.
            grouped_data = self.df.groupby(self.group_by_cols, dropna=False)
        except KeyError as e:
            # This should be caught by the check above, but as a safeguard:
            raise ValueError(f"Error during groupby operation. Ensure all group_by_cols exist: {e}")

        all_grouped_metrics_results: list[dict[str, Any]] = []

        for group_name_tuple, group_df in grouped_data:
            if group_df.empty:
                continue # Skip empty groups

            group_values_dict: dict[str, Any] = {}
            if isinstance(group_name_tuple, tuple):
                group_values_dict = dict(zip(self.group_by_cols, group_name_tuple, strict=False))
            else: # Single grouping column
                group_values_dict = {self.group_by_cols[0]: group_name_tuple}

            # Create a temporary calculator for this specific group_df
            # Do not pass group_by_cols to this temp instance, as we want its overall metrics.
            temp_calculator = MetricsCalculator(group_df.copy())
            group_metrics = temp_calculator.calculate_all_metrics()

            all_grouped_metrics_results.append({
                "group_values": group_values_dict,
                "metrics": group_metrics
            })

        return all_grouped_metrics_results

    def calculate_per_level_metrics(self, variable: str) -> pd.DataFrame:
        """
        Computes metrics for each unique level of a given variable (column).

        Args:
            variable (str): The column name to group by (e.g., a variable_to_test).

        Returns:
            pd.DataFrame: DataFrame with one row per level, columns include:
                - variable (the level value)
                - mae, nmae, mse, rmse, mean_accuracy, std_accuracy, count, etc.
        """
        if variable not in self.df.columns:
            raise ValueError(f"Column '{variable}' not found in DataFrame.")

        results = []
        for level, group_df in self.df.groupby(variable, dropna=False):
            if group_df.empty:
                continue
            # Only use numeric preds/targets for stats
            preds = pd.to_numeric(group_df['preds'], errors='coerce')
            targets = pd.to_numeric(group_df['targets'], errors='coerce')
            stats = compute_regression_stats(preds, targets)
            row = {variable: level}
            row.update(stats)
            results.append(row)
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values(by=variable).reset_index(drop=True)
        return df

    def calculate_cross_variable_stats(self, variable1: str, variable2: str) -> pd.DataFrame:
        """
        Computes regression statistics for each unique combination of levels from two variables.

        Args:
            variable1 (str): The first column name to group by.
            variable2 (str): The second column name to group by.

        Returns:
            pd.DataFrame: DataFrame with each row representing a unique combination of
                          variable1 and variable2 levels. Columns include variable1, variable2,
                          and all metrics from compute_regression_stats (mae, mse, count, etc.).

        Raises:
            ValueError: If variable1 or variable2 are not in the DataFrame.
        """
        if variable1 not in self.df.columns:
            raise ValueError(f"Column '{variable1}' not found in DataFrame.")
        if variable2 not in self.df.columns:
            raise ValueError(f"Column '{variable2}' not found in DataFrame.")
        if variable1 == variable2:
            raise ValueError("variable1 and variable2 must be different column names.")

        results = []
        # dropna=False in groupby includes combinations where one or both variable levels might be NaN
        for (level1, level2), group_df in self.df.groupby([variable1, variable2], dropna=False):
            if group_df.empty:
                continue

            preds = pd.to_numeric(group_df['preds'], errors='coerce')
            targets = pd.to_numeric(group_df['targets'], errors='coerce')

            stats = compute_regression_stats(preds, targets) # This is the global function

            row = {variable1: level1, variable2: level2}
            row.update(stats)
            results.append(row)

        cross_stats_df = pd.DataFrame(results)

        # Attempt to sort by the variable columns for consistent output, handling potential mixed types
        if not cross_stats_df.empty:
            try:
                cross_stats_df = cross_stats_df.sort_values(by=[variable1, variable2]).reset_index(drop=True)
            except TypeError:
                # If direct sorting fails (e.g. mixed types that pandas can't sort across columns easily without conversion),
                # fallback to sorting by string representation or just reset index.
                # For simplicity here, just reset index if sort fails.
                print(f"Warning: Could not sort cross_stats_df by '{variable1}' and '{variable2}' due to type issues. Proceeding unsorted.")
                cross_stats_df = cross_stats_df.reset_index(drop=True)

        return cross_stats_df

if __name__ == '__main__':
    # Example Usage:
    console = pd.Series # Dummy for rich console if not available
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
    except ImportError:
        print("Rich library not found. Plain print will be used for output.")

    def display_results(metrics_dict: dict[str, Any], title: str, grouped_metrics_list: list[dict[str, Any]] | None = None):
        if isinstance(console, pd.Series): # rich not available
            print(f"\\n--- {title} ---")
            import json
            print("Overall Metrics:")
            print(json.dumps(metrics_dict, indent=2, default=str))
            if grouped_metrics_list:
                print("\\nGrouped Metrics:")
                print(json.dumps(grouped_metrics_list, indent=2, default=str))
            return

        console.print(f"\\n[bold cyan]--- {title} ---[/bold cyan]")
        console.print("[u]Overall Metrics:[/u]")

        type_info = metrics_dict.get("target_type_info", {})
        console.print("[bold]Target Type Analysis:[/bold]")
        console.print(f"  Type: {type_info.get('type')}")
        console.print(f"  Dimensions: {type_info.get('dimensions')}")
        console.print(f"  Is Numeric Tuple: {type_info.get('is_numeric_tuple')}")

        acc_metrics = metrics_dict.get("accuracy_metrics")
        if acc_metrics:
            console.print("\\n[bold]Accuracy Metrics:[/bold]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="dim")
            table.add_column("Value")
            table.add_row("Mean Accuracy", f"{acc_metrics.get('mean_accuracy', np.nan):.4f}")
            table.add_row("Std Accuracy (Indicator)", f"{acc_metrics.get('std_accuracy', np.nan):.4f}")
            table.add_row("Count", str(acc_metrics.get('count', 0)))
            console.print(table)

        num_metrics = metrics_dict.get("numeric_metrics")
        if num_metrics:
            console.print("\\n[bold]Numeric Metrics:[/bold]")
            if isinstance(num_metrics, list): # Tuple metrics
                for i, dim_metrics in enumerate(num_metrics):
                    console.print(f"  [italic]Dimension {i}:[/italic]")
                    dim_table = Table(show_header=True, header_style="bold magenta")
                    dim_table.add_column("Metric", style="dim")
                    dim_table.add_column("Value")
                    dim_table.add_row("MAE", f"{dim_metrics.get('mae', np.nan):.4f}")
                    dim_table.add_row("NMAE", f"{dim_metrics.get('nmae', np.nan):.4f}")
                    dim_table.add_row("MSE", f"{dim_metrics.get('mse', np.nan):.4f}")
                    dim_table.add_row("RMSE", f"{dim_metrics.get('rmse', np.nan):.4f}")
                    dim_table.add_row("Count", str(dim_metrics.get('count', 0)))
                    if "dimension_error" in dim_metrics:
                         dim_table.add_row("Error", dim_metrics["dimension_error"])
                    console.print(dim_table)
            elif isinstance(num_metrics, dict): # Single numeric metrics
                num_table = Table(show_header=True, header_style="bold magenta")
                num_table.add_column("Metric", style="dim")
                num_table.add_column("Value")
                num_table.add_row("MAE", f"{num_metrics.get('mae', np.nan):.4f}")
                num_table.add_row("NMAE", f"{num_metrics.get('nmae', np.nan):.4f}")
                num_table.add_row("MSE", f"{num_metrics.get('mse', np.nan):.4f}")
                num_table.add_row("RMSE", f"{num_metrics.get('rmse', np.nan):.4f}")
                num_table.add_row("Count", str(num_metrics.get('count', 0)))
                console.print(num_table)

        if grouped_metrics_list:
            console.print("\n[u]Grouped Metrics:[/u]")
            if not grouped_metrics_list:
                console.print("  No groups to display or group_by_cols not specified.")
                return

            for item in grouped_metrics_list:
                group_vals_str = ", ".join([f"{k}='{v}'" for k, v in item['group_values'].items()])
                console.print(f"\n  [bold yellow]Group: {group_vals_str}[/bold yellow]")

                group_type_info = item['metrics'].get("target_type_info", {})
                console.print(f"    Target Type: {group_type_info.get('type')}, Dims: {group_type_info.get('dimensions')}, Numeric Tuple: {group_type_info.get('is_numeric_tuple')}")
                console.print(f"    Count for this group: {item['metrics'].get('accuracy_metrics', {}).get('count', 0)}")

                acc_metrics_group = item['metrics'].get("accuracy_metrics")
                if acc_metrics_group and acc_metrics_group.get('count', 0) > 0:
                    console.print("    [bold]Accuracy Metrics (Group):[/bold]")
                    acc_table_group = Table(show_header=True, header_style="bold magenta")
                    acc_table_group.add_column("Metric", style="dim")
                    acc_table_group.add_column("Value")
                    acc_table_group.add_row("Mean Accuracy", f"{acc_metrics_group.get('mean_accuracy', np.nan):.4f}")
                    acc_table_group.add_row("Std Accuracy (Indicator)", f"{acc_metrics_group.get('std_accuracy', np.nan):.4f}")
                    console.print(acc_table_group)

                num_metrics_group = item['metrics'].get("numeric_metrics")
                if num_metrics_group:
                    console.print("    [bold]Numeric Metrics (Group):[/bold]")
                    if isinstance(num_metrics_group, list): # Tuple metrics for the group
                        for i, dim_metrics in enumerate(num_metrics_group):
                            if dim_metrics.get('count', 0) > 0:
                                console.print(f"      [italic]Dimension {i}:[/italic]")
                                dim_table_group = Table(show_header=True, header_style="bold magenta")
                                dim_table_group.add_column("Metric", style="dim")
                                dim_table_group.add_column("Value")
                                dim_table_group.add_row("MAE", f"{dim_metrics.get('mae', np.nan):.4f}")
                                dim_table_group.add_row("NMAE", f"{dim_metrics.get('nmae', np.nan):.4f}")
                                dim_table_group.add_row("MSE", f"{dim_metrics.get('mse', np.nan):.4f}")
                                dim_table_group.add_row("RMSE", f"{dim_metrics.get('rmse', np.nan):.4f}")
                                console.print(dim_table_group)
                    elif isinstance(num_metrics_group, dict) and num_metrics_group.get('count',0) > 0: # Single numeric metrics for the group
                        num_table_group = Table(show_header=True, header_style="bold magenta")
                        num_table_group.add_column("Metric", style="dim")
                        num_table_group.add_column("Value")
                        num_table_group.add_row("MAE", f"{num_metrics_group.get('mae', np.nan):.4f}")
                        num_table_group.add_row("NMAE", f"{num_metrics_group.get('nmae', np.nan):.4f}")
                        num_table_group.add_row("MSE", f"{num_metrics_group.get('mse', np.nan):.4f}")
                        num_table_group.add_row("RMSE", f"{num_metrics_group.get('rmse', np.nan):.4f}")
                        console.print(num_table_group)

    # --- Test Cases ---
    # 1. Numeric Data
    data_numeric = {
        'preds': [1, 2, 3, 4, 5.5, 6, 7, 8, 9, 10],
        'targets': [1.1, 2.3, 2.9, 4.2, 5.0, 6.5, 6.8, 8.3, 9.1, 9.8]
    }
    df_numeric = pd.DataFrame(data_numeric)
    calc_numeric = MetricsCalculator(df_numeric)
    results_numeric = calc_numeric.calculate_all_metrics()
    display_results(results_numeric, "Numeric Data Metrics")

    # 2. String Data
    data_string = {
        'preds': ["apple", "banana", "cherry", "apple", "banana"],
        'targets': ["apple", "orange", "cherry", "apple", "orange"]
    }
    df_string = pd.DataFrame(data_string)
    calc_string = MetricsCalculator(df_string)
    results_string = calc_string.calculate_all_metrics()
    display_results(results_string, "String Data Metrics")

    # 3. Tuple Data (Numeric)
    data_tuple_numeric = {
        'preds': [(1, 10), (2, 22), (3, 28), (4, 40), (5, 52)],
        'targets': [(1.1, 12), (2.3, 20), (2.9, 30), (4.2, 38), (5.0, 50)]
    }
    df_tuple_numeric = pd.DataFrame(data_tuple_numeric)
    calc_tuple_numeric = MetricsCalculator(df_tuple_numeric)
    results_tuple_numeric = calc_tuple_numeric.calculate_all_metrics()
    display_results(results_tuple_numeric, "Numeric Tuple Data Metrics")

    # 4. Mixed Tuple Data (Non-Numeric elements)
    data_tuple_mixed = {
        'preds': [(1, "a"), ("b", 20), (3, "c")],
        'targets': [(1, "a"), ("x", 20), (3, "y")]
    }
    df_tuple_mixed = pd.DataFrame(data_tuple_mixed)
    calc_tuple_mixed = MetricsCalculator(df_tuple_mixed)
    results_tuple_mixed = calc_tuple_mixed.calculate_all_metrics()
    display_results(results_tuple_mixed, "Mixed Tuple Data Metrics")

    # 5. Data with NaNs
    data_nans = {
        'preds': [1, np.nan, 3, 4, np.nan, 6],
        'targets': [1.1, 2.3, np.nan, 4.2, 5.0, 6.5]
    }
    df_nans = pd.DataFrame(data_nans)
    calc_nans = MetricsCalculator(df_nans)
    results_nans = calc_nans.calculate_all_metrics()
    display_results(results_nans, "Data with NaNs Metrics")

    # 6. Data with pd.NA
    data_pd_na = {
        'preds': [1, pd.NA, 3, "test", pd.NA],
        'targets': [1.0, 2.0, pd.NA, "test", "true_val"]
    }
    df_pd_na = pd.DataFrame(data_pd_na, dtype="object") # Use object dtype for pd.NA compatibility
    calc_pd_na = MetricsCalculator(df_pd_na)
    results_pd_na = calc_pd_na.calculate_all_metrics()
    display_results(results_pd_na, "Data with pd.NA Metrics")

    # 7. Empty DataFrame
    df_empty = pd.DataFrame({'preds': [], 'targets': []})
    calc_empty = MetricsCalculator(df_empty)
    results_empty = calc_empty.calculate_all_metrics()
    display_results(results_empty, "Empty DataFrame Metrics")

    # 8. All NaNs in targets
    data_all_nan_targets = {
        'preds': [1, 2, 3],
        'targets': [np.nan, pd.NA, None]
    }
    df_all_nan_targets = pd.DataFrame(data_all_nan_targets, dtype="object")
    calc_all_nan_targets = MetricsCalculator(df_all_nan_targets)
    results_all_nan_targets = calc_all_nan_targets.calculate_all_metrics()
    display_results(results_all_nan_targets, "All NaN/NA/None Targets Metrics")

    # 9. Numeric data where target range is 0
    data_zero_range = {
        'preds': [5, 5.1, 4.9, 5.05],
        'targets': [5, 5, 5, 5]
    }
    df_zero_range = pd.DataFrame(data_zero_range)
    calc_zero_range = MetricsCalculator(df_zero_range)
    results_zero_range = calc_zero_range.calculate_all_metrics()
    display_results(results_zero_range, "Numeric Data (Zero Target Range) Metrics")

    # 10. Numeric data where target mean and range is 0
    data_zero_mean_range = {
        'preds': [0, 0.1, -0.1, 0.05],
        'targets': [0, 0, 0, 0]
    }
    df_zero_mean_range = pd.DataFrame(data_zero_mean_range)
    calc_zero_mean_range = MetricsCalculator(df_zero_mean_range)
    results_zero_mean_range = calc_zero_mean_range.calculate_all_metrics()
    display_results(results_zero_mean_range, "Numeric Data (Zero Target Mean & Range) Metrics")

    # 11. Numeric tuple with inconsistent lengths or non-numeric elements mid-data
    # Note: Current _analyze_target_type bases on first valid target.
    # calculate_numeric_metrics has some robustness for this.
    data_tuple_inconsistent = {
        'preds': [(1, 10), (2, "error"), (3,30,99), (4,)], # Mix of valid, wrong type, wrong length
        'targets': [(1.1, 12), (2.3, 20), (2.9, 30), (4.2, 40)] # Assume analyzer picks (float, float) dim 2
    }
    df_tuple_inconsistent = pd.DataFrame(data_tuple_inconsistent)
    calc_tuple_inconsistent = MetricsCalculator(df_tuple_inconsistent)
    results_tuple_inconsistent = calc_tuple_inconsistent.calculate_all_metrics()
    display_results(results_tuple_inconsistent, "Inconsistent Numeric Tuple Data Metrics")

    # --- Test Cases for Grouped Metrics ---
    data_grouped = {
        'preds':   [1,    2,    3,    4,    5,    6,    1.5,  2.5,  3.5,  4.5,  5.5,  6.5],
        'targets': [1.1,  2.2,  2.8,  4.3,  4.8,  6.2,  1.0,  2.1,  2.9,  4.0,  5.1,  6.0],
        'var1':    ['A',  'A',  'A',  'B',  'B',  'B',  'A',  'A',  'B',  'B',  'A',  'A'],
        'var2':    [100,  100,  200,  200,  100,  100,  100,  200,  200,  100,  100,  200]
    }
    df_grouped = pd.DataFrame(data_grouped)

    # Calculate with grouping
    group_cols = ['var1', 'var2']
    calc_grouped = MetricsCalculator(df_grouped, group_by_cols=group_cols)
    overall_results_grouped_df = calc_grouped.calculate_all_metrics() # Overall metrics for the df with group_by_cols
    grouped_metrics_list = calc_grouped.calculate_grouped_metrics()
    display_results(overall_results_grouped_df, "Grouped Data Example (var1, var2)", grouped_metrics_list)

    # Calculate with a single grouping column
    group_cols_single = ['var1']
    calc_grouped_single = MetricsCalculator(df_grouped, group_by_cols=group_cols_single)
    overall_results_single_group = calc_grouped_single.calculate_all_metrics()
    grouped_metrics_list_single = calc_grouped_single.calculate_grouped_metrics()
    display_results(overall_results_single_group, "Grouped Data Example (var1 only)", grouped_metrics_list_single)

    # Calculate with group_by_cols not provided (should only show overall)
    calc_no_group = MetricsCalculator(df_grouped) # No group_by_cols
    overall_results_no_group = calc_no_group.calculate_all_metrics()
    grouped_metrics_no_group = calc_no_group.calculate_grouped_metrics() # Should be empty
    display_results(overall_results_no_group, "No Grouping Example", grouped_metrics_no_group)


    # practictal test with diagnostic_results/diagnostic_evaluation_results_gemma_20250508_004823.csv
    df_diagnostic = pd.read_csv("diagnostic_results/diagnostic_evaluation_results_gemma_20250508_120051.csv")

    group_cols = ['number']

    calc_diagnostic = MetricsCalculator(df_diagnostic, group_by_cols=group_cols)

    results_diagnostic = calc_diagnostic.calculate_all_metrics()
    grouped_metrics_list_diagnostic = calc_diagnostic.calculate_grouped_metrics()
    display_results(results_diagnostic, "Diagnostic Evaluation Results")

    # For displaying only grouped metrics, pass an empty dict for the first argument
    # and the actual list of grouped metrics for the third argument.
    display_results({}, "Diagnostic Evaluation Results Grouped by number", grouped_metrics_list_diagnostic)

    calc = MetricsCalculator(df_diagnostic, group_by_cols=['number'])
    mae_per_number = calc.calculate_per_level_metrics('number')
    print(mae_per_number)

    # Choose the variable to test (e.g., "number" or "blur")
    for variable in ["number"]:
        print(f"\nPer-level metrics for variable: {variable}")
        calc_diagnostic = MetricsCalculator(df_diagnostic)
        per_level_df = calc_diagnostic.calculate_per_level_metrics(variable)
        print(per_level_df)
