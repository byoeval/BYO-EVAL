import os  # Added for directory creation
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from evaluation_pipeline.metrics_calculator import MetricsCalculator

# Removed compute_regression_stats as it's used internally by MetricsCalculator's methods primarily

class MetricsPlotter:
    """
    A class for plotting various metrics against levels of a specified variable.

    The input DataFrame should contain 'preds' (predictions) and 'targets' (true values)
    columns, along with a specified 'variable_col' whose levels will form the x-axis
    of the plots.
    """

    def __init__(self, df: pd.DataFrame, variable_col: str, output_dir: str | None = "plots"):
        """
        Initializes the MetricsPlotter.

        Args:
            df (pd.DataFrame): The input DataFrame. Must contain 'preds' and 'targets'
                               columns (or 'pred' and 'target' which will be renamed),
                               and the 'variable_col'.
            variable_col (str): The name of the column in 'df' that contains the
                                variable whose levels will be used for the x-axis.
            output_dir (Optional[str]): The directory where plots will be saved.
                                        If None, saving capabilities might be disabled or warn.
                                        Defaults to "plots". The directory will be created if it doesn't exist.

        Raises:
            TypeError: If 'df' is not a pandas DataFrame.
            ValueError: If required columns ('preds', 'targets', 'variable_col')
                        are missing from 'df'.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input df must be a pandas DataFrame.")

        self.df_original = df.copy() # Store the original DataFrame to be used by suite methods
        self._initialize_for_variable(variable_col, output_dir)

    def _initialize_for_variable(self, variable_col: str, output_dir: str | None):
        """
        Internal helper to set up the plotter for a specific variable_col and output_dir.
        This allows re-initialization or fresh setup for different variables from the same original df.
        """
        self.df = self.df_original.copy() # Work with a copy for this specific initialization
        self.variable_col = variable_col
        self.output_dir = output_dir

        if self.output_dir:
            try:
                os.makedirs(self.output_dir, exist_ok=True)
                # print(f"Output directory '{self.output_dir}' ensured for plotter instance.") # Can be verbose
            except Exception as e:
                print(f"Error creating output directory '{self.output_dir}' in plotter: {e}")
                # For suite operations, we might not want to raise here, but rely on save_figure to fail.

        # Standardize pred/target column names in self.df
        if 'pred' in self.df.columns and 'target' in self.df.columns:
            self.df.rename(columns={'pred': 'preds', 'target': 'targets'}, inplace=True)
        elif 'pred' in self.df.columns and 'preds' not in self.df.columns:
            self.df.rename(columns={'pred': 'preds'}, inplace=True)
        elif 'target' in self.df.columns and 'targets' not in self.df.columns:
            self.df.rename(columns={'target': 'targets'}, inplace=True)

        if 'preds' not in self.df.columns:
            raise ValueError("DataFrame must contain 'preds' column after renaming attempt.")
        if 'targets' not in self.df.columns:
            raise ValueError("DataFrame must contain 'targets' column after renaming attempt.")
        if self.variable_col not in self.df.columns:
            # This check is important. If variable_col is not in df, subsequent ops will fail.
            raise ValueError(f"Variable column '{self.variable_col}' not found in DataFrame.")

        # Ensure 'preds' and 'targets' are numeric for direct calculations if needed
        try:
            self.df['preds'] = pd.to_numeric(self.df['preds'], errors='coerce')
            self.df['targets'] = pd.to_numeric(self.df['targets'], errors='coerce')
        except Exception as e:
            print(f"Warning: Could not convert 'preds' or 'targets' to numeric directly in plotter: {e}")

        all_unique_levels = []
        if self.variable_col in self.df.columns: # Should always be true due to check above
            # print(f"DEBUG: Initial unique values from df[{self.variable_col}].dropna().unique(): {self.df[self.variable_col].dropna().unique()}")
            all_unique_levels = sorted(self.df[self.variable_col].dropna().unique())
            # print(f"DEBUG: Initial all_unique_levels (sorted): {all_unique_levels}")
            try:
                temp_series = pd.to_numeric(pd.Series(all_unique_levels), errors='coerce')
                if not temp_series.isna().all():
                    all_unique_levels = sorted(temp_series.dropna().unique())
                    # print(f"DEBUG: all_unique_levels after potential numeric conversion: {all_unique_levels}")
            except ValueError:
                # print(f"Warning: Variable column '{self.variable_col}' levels used as is, as numeric conversion was not fully applicable.")
                pass # Keep original sorting if not fully numeric

        if not all_unique_levels:
            # print(f"Warning: No unique, non-NaN levels found for variable '{self.variable_col}'. Plots might be empty or incorrect.")
            self.per_level_stats_df = pd.DataFrame(columns=[self.variable_col])
        else:
            calculator = MetricsCalculator(self.df.copy())
            self.per_level_stats_df = calculator.calculate_per_level_metrics(self.variable_col)
            # print(f"DEBUG: self.per_level_stats_df after MetricsCalculator.calculate_per_level_metrics:")
            # print(self.per_level_stats_df.head())

            if self.per_level_stats_df.empty and all_unique_levels:
                self.per_level_stats_df = pd.DataFrame({self.variable_col: all_unique_levels})
                # print(f"DEBUG: self.per_level_stats_df created as placeholder with all_unique_levels (empty from calculator):")
                # print(self.per_level_stats_df.head())

            if self.variable_col in self.per_level_stats_df.columns:
                # print(f"DEBUG: '{self.variable_col}' found in self.per_level_stats_df. Attempting reindex.")
                try:
                    if pd.Series(all_unique_levels).dtype.kind in 'ifc':
                        self.per_level_stats_df[self.variable_col] = pd.to_numeric(self.per_level_stats_df[self.variable_col], errors='coerce')
                        self.per_level_stats_df.dropna(subset=[self.variable_col], inplace=True)
                except Exception:
                    # print(f"Warning: Type conversion for '{self.variable_col}' in per_level_stats_df failed before reindex: {e}")
                    pass

                self.per_level_stats_df = self.per_level_stats_df.set_index(self.variable_col)
                self.per_level_stats_df = self.per_level_stats_df.reindex(all_unique_levels)
                self.per_level_stats_df = self.per_level_stats_df.reset_index()

                if 'index' in self.per_level_stats_df.columns and self.variable_col not in self.per_level_stats_df.columns:
                    self.per_level_stats_df.rename(columns={'index': self.variable_col}, inplace=True)
                elif 'level_0' in self.per_level_stats_df.columns and self.variable_col not in self.per_level_stats_df.columns:
                    self.per_level_stats_df.rename(columns={'level_0': self.variable_col}, inplace=True)

                if self.variable_col in self.per_level_stats_df.columns:
                    try:
                        if pd.Series(all_unique_levels).dtype.kind in 'ifc':
                             self.per_level_stats_df[self.variable_col] = pd.to_numeric(self.per_level_stats_df[self.variable_col], errors='coerce')
                        self.per_level_stats_df = self.per_level_stats_df.sort_values(by=self.variable_col).reset_index(drop=True)
                    except Exception:
                        # print(f"Warning: Could not sort per_level_stats_df by '{self.variable_col}' after reindexing: {e}")
                        pass
                # else:
                #      print(f"Warning: '{self.variable_col}' column is missing after reindexing operations.")
            elif all_unique_levels:
                 self.per_level_stats_df = pd.DataFrame({self.variable_col: all_unique_levels})
                 if pd.Series(all_unique_levels).dtype.kind in 'ifc':
                      self.per_level_stats_df[self.variable_col] = pd.to_numeric(self.per_level_stats_df[self.variable_col], errors='coerce')
                 self.per_level_stats_df = self.per_level_stats_df.sort_values(by=self.variable_col).reset_index(drop=True)

        if self.per_level_stats_df.empty or self.variable_col not in self.per_level_stats_df.columns:
            # print(f"Warning: '{self.variable_col}' still not properly established in per_level_stats_df.")
            if all_unique_levels:
                self.per_level_stats_df = pd.DataFrame({self.variable_col: all_unique_levels})
                for stat_col in ['mae', 'mae_std', 'mean_accuracy', 'std_accuracy', 'pred_mean', 'pred_std', 'target_mean', 'count']:
                    if stat_col not in self.per_level_stats_df.columns:
                        self.per_level_stats_df[stat_col] = np.nan
            else:
                self.per_level_stats_df = pd.DataFrame(columns=[self.variable_col, 'mae', 'mae_std', 'mean_accuracy', 'std_accuracy', 'pred_mean', 'pred_std', 'target_mean', 'count'])

        # print(f"DEBUG: Final self.per_level_stats_df in _initialize_for_variable ({self.variable_col}):")
        # print(self.per_level_stats_df[[self.variable_col] + [col for col in ['mae', 'count'] if col in self.per_level_stats_df.columns]].to_string())
        # print(f"DEBUG: Data types of final self.per_level_stats_df in _initialize_for_variable:")
        # print(self.per_level_stats_df.dtypes)

    def _prepare_ax(self, ax: plt.Axes | None = None) -> plt.Axes:
        """Prepares Axes object for plotting."""
        if ax is None:
            _, ax = plt.subplots()
        return ax

    def _set_plot_fonts(self, ax, title=None, title_fontsize=30, axis_fontsize=25, tick_fontsize=17):
        """Helper method to set consistent font sizes across plots."""
        if title:
            ax.set_title(title, fontsize=title_fontsize)
        else:
            # Set font size for existing title if any
            ax.set_title(ax.get_title(), fontsize=title_fontsize)

        # Set font sizes for axis labels
        ax.set_xlabel(ax.get_xlabel(), fontsize=axis_fontsize)
        ax.set_ylabel(ax.get_ylabel(), fontsize=axis_fontsize)

        # Set font sizes for tick labels
        ax.tick_params(axis='both', labelsize=tick_fontsize)

        # Set font size for legend if it exists
        legend = ax.get_legend()
        if legend is not None:
            # Check if legend is exterior (outside plot area) or interior
            if legend._loc in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:  # Exterior legend
                for text in legend.get_texts():
                    text.set_fontsize(30)
            else:  # Interior legend
                for text in legend.get_texts():
                    text.set_fontsize(25)

        # Ensure all text elements in the plot have appropriate font sizes
        for text_obj in ax.texts:
            # Text within the plot area (like annotations) should use the axis_fontsize
            text_obj.set_fontsize(axis_fontsize)

        # Update any additional text artists that might be in the axes
        for child in ax.get_children():
            if isinstance(child, plt.Text):
                # Location-based font sizing
                if child.get_position()[1] > 0.9:  # If near the top, likely a title
                    child.set_fontsize(title_fontsize)
                    #child.set_fontweight('bold')
                else:  # Other text elements
                    child.set_fontsize(axis_fontsize)

    def _handle_x_ticks(self, ax):
        """Helper method to handle x-tick display for plots with many ticks.

        If there are more than 10 x-ticks, only display text for every other x-tick
        starting from the 10th tick (i.e., show 10, 12, 14, etc. but hide 11, 13, 15, etc.)
        """
        # Get current x-tick positions and labels
        x_ticks = ax.get_xticks()
        x_tick_labels = [label.get_text() for label in ax.get_xticklabels()]

        # If we have more than 10 ticks
        if len(x_ticks) > 10:
            # Create a mask where we keep all ticks below 10, and then every other tick for the rest
            mask = np.array([(i < 10 or (i >= 10 and (i % 2 == 0))) for i in range(len(x_ticks))])

            # Apply the mask to the tick labels (replace non-masked labels with empty strings)
            new_labels = [label if mask[i] else '' for i, label in enumerate(x_tick_labels)]

            # Apply the new labels
            ax.set_xticklabels(new_labels)

    def plot_mae(self,
                 ax: plt.Axes | None = None,
                 show_std_fill: bool = False,
                 show_abs_error_boxplot: bool = False,
                 abs_error_quantiles_fill: tuple[float, float] | None = None,
                 plot_kwargs: dict[str, Any] | None = None,
                 std_fill_kwargs: dict[str, Any] | None = None,
                 boxplot_kwargs: dict[str, Any] | None = None,
                 quantile_fill_kwargs: dict[str, Any] | None = None
                 ) -> plt.Axes:
        """
        Plots Mean Absolute Error (MAE) against the levels of 'variable_col'.

        Args:
            ax (Optional[plt.Axes]): Matplotlib Axes object to plot on. If None, creates a new one.
            show_std_fill (bool): If True, shows MAE +/- std(absolute errors) as a filled area.
                                  'mae' and 'mae_std' columns must be in per_level_stats_df.
            show_abs_error_boxplot (bool): If True, displays boxplots of absolute errors for each level.
                                           This will use the raw 'preds' and 'targets' from the input df.
            abs_error_quantiles_fill (Optional[Tuple[float, float]]): Tuple (q_low, q_high), e.g., (0.25, 0.75).
                                                                   If provided, fills area between these quantiles
                                                                   of absolute errors.
            plot_kwargs (Optional[Dict[str, Any]]): Keyword arguments for the main MAE line plot.
            std_fill_kwargs (Optional[Dict[str, Any]]): Keyword arguments for the std deviation fill_between.
            boxplot_kwargs (Optional[Dict[str, Any]]): Keyword arguments for the seaborn boxplot.
            quantile_fill_kwargs (Optional[Dict[str, Any]]): Keyword arguments for the quantile fill_between.

        Returns:
            plt.Axes: The Axes object with the plot.

        Raises:
            ValueError: If required columns for plotting are missing from per_level_stats_df
                        or if 'abs_error_quantiles_fill' provides invalid quantiles.
        """
        ax = self._prepare_ax(ax)
        if self.per_level_stats_df.empty or self.variable_col not in self.per_level_stats_df.columns:
            ax.text(0.5, 0.5, "No data to plot for MAE.", ha='center', va='center')
            ax.set_title(f"MAE vs {self.variable_col}")
            return ax


        if 'mae' not in self.per_level_stats_df.columns:
            raise ValueError("Column 'mae' not found in per_level_stats_df. Cannot plot MAE.")

        # x_values are the actual category labels (e.g., [1, 2, ..., 10])
        x_categories = self.per_level_stats_df[self.variable_col].tolist()
        y_mae = self.per_level_stats_df['mae'].tolist()

        # positions are 0-indexed for plotting (e.g., [0, 1, ..., 9])
        x_positions = list(range(len(x_categories)))



        _plot_kwargs = {'label': 'MAE', 'marker': 'o'}
        if plot_kwargs: _plot_kwargs.update(plot_kwargs)
        ax.plot(x_positions, y_mae, **_plot_kwargs) # Use positions for line plot

        if show_std_fill:
            if 'mae_std' not in self.per_level_stats_df.columns:
                raise ValueError("Column 'mae_std' not found in per_level_stats_df. Cannot show std fill.")
            mae_std_values = self.per_level_stats_df['mae_std'].tolist()
            _std_fill_kwargs = {'alpha': 0.2, 'label': 'MAE +/- Std(Abs Errors)'}
            if std_fill_kwargs: _std_fill_kwargs.update(std_fill_kwargs)
            ax.fill_between(x_positions,
                            [m - s for m, s in zip(y_mae, mae_std_values, strict=False)],
                            [m + s for m, s in zip(y_mae, mae_std_values, strict=False)],
                            **_std_fill_kwargs) # Use positions

        temp_df = self.df.copy()
        preds_numeric = pd.to_numeric(temp_df['preds'], errors='coerce')
        targets_numeric = pd.to_numeric(temp_df['targets'], errors='coerce')

        valid_idx = preds_numeric.notna() & targets_numeric.notna()
        temp_df['abs_error'] = np.nan
        if valid_idx.any():
            temp_df.loc[valid_idx, 'abs_error'] = (preds_numeric[valid_idx] - targets_numeric[valid_idx]).abs()

        plot_df_abs_error = temp_df.dropna(subset=['abs_error', self.variable_col])

        if show_abs_error_boxplot:
            if not plot_df_abs_error.empty:
                _boxplot_kwargs = {
                    'medianprops': {'color': 'black', 'linewidth': 1.5},
                    'boxprops': {'facecolor': 'none', 'edgecolor': 'black'}
                }
                if boxplot_kwargs: _boxplot_kwargs.update(boxplot_kwargs)
                try:

                    sns.boxplot(x=self.variable_col, y='abs_error', data=plot_df_abs_error, ax=ax, order=x_categories, **_boxplot_kwargs)
                except Exception as e:
                    print(f"Could not generate boxplot: {e}")
            else:
                print("Not enough data for absolute error boxplot after filtering NaNs.")

        if abs_error_quantiles_fill:
            if not (isinstance(abs_error_quantiles_fill, tuple) and len(abs_error_quantiles_fill) == 2 and
                    0 <= abs_error_quantiles_fill[0] < abs_error_quantiles_fill[1] <= 1):
                raise ValueError("abs_error_quantiles_fill must be a tuple of two floats (q_low, q_high) between 0 and 1.")

            if not plot_df_abs_error.empty:
                q_low, q_high = abs_error_quantiles_fill
                grouped_abs_errors = plot_df_abs_error.groupby(self.variable_col)['abs_error']
                try:
                    quantiles = grouped_abs_errors.quantile([q_low, q_high])
                    if not quantiles.empty:
                        quantiles_df = quantiles.unstack()
                        # Reindex quantiles_df to align with x_categories, then use x_positions for plotting
                        if not quantiles_df.empty and x_categories:
                            # Ensure quantiles_df.index matches the type of x_categories for reindexing
                            idx_type = pd.Series(x_categories).dtype
                            try:
                                quantiles_df.index = quantiles_df.index.astype(idx_type)
                            except Exception as e_conv:
                                print(f"Warning: could not convert quantiles_df.index to type {idx_type}: {e_conv}")

                            quantiles_df = quantiles_df.reindex(x_categories)

                            _quantile_fill_kwargs = {'alpha': 0.3, 'label': f'Abs Error {q_low*100:.0f}-{q_high*100:.0f}th pct'}
                            if quantile_fill_kwargs: _quantile_fill_kwargs.update(quantile_fill_kwargs)

                            valid_quantile_idx = quantiles_df[q_low].notna() & quantiles_df[q_high].notna()
                            if valid_quantile_idx.any():
                                # Use x_positions for the fill_between x-coordinates
                                ax.fill_between([x_positions[i] for i, valid in enumerate(valid_quantile_idx) if valid],
                                                quantiles_df[q_low][valid_quantile_idx],
                                                quantiles_df[q_high][valid_quantile_idx],
                                                **_quantile_fill_kwargs)
                            else:
                                print("No valid quantile data to plot after alignment/filtering.")
                        else:
                             print("Quantile data frame is empty or x_categories are empty, cannot plot quantile fill.")
                    else:
                        print("No quantile data computed.")
                except Exception as e:
                    print(f"Could not generate quantile fill for absolute errors: {e}")
            else:
                print("Not enough data for absolute error quantile fill after filtering NaNs.")

        ax.set_xlabel(self.variable_col)
        ax.set_ylabel("MAE")
        ax.set_title(f"MAE vs {self.variable_col}")

        # Set xticks and labels
        if x_positions:
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_categories)
            ax.set_xlim(x_positions[0] - 0.5, x_positions[-1] + 0.5)

            # Handle x-tick display for plots with many ticks
            if len(x_positions) > 10:
                # Get current tick labels
                x_tick_labels = [label.get_text() for label in ax.get_xticklabels()]

                # Create a mask where we show all ticks below index 10, then only even-indexed ticks after that
                mask = [(i < 10 or (i >= 10 and (i % 2 == 1))) for i in range(len(x_positions))]

                # Apply the mask to create new labels
                new_labels = [label if mask[i] else '' for i, label in enumerate(x_tick_labels)]

                # Set the new labels
                ax.set_xticklabels(new_labels)

        # After all plotting is done, add this line at the end before returning:
        self._set_plot_fonts(ax, title=f"MAE vs {self.variable_col}")

        if _plot_kwargs.get('label') or (show_std_fill and _std_fill_kwargs.get('label')) or \
           (abs_error_quantiles_fill and not plot_df_abs_error.empty and _quantile_fill_kwargs.get('label')):
            ax.legend()
            self._set_plot_fonts(ax)
        return ax

    def plot_nmae(self,
                 ax: plt.Axes | None = None,
                 show_std_fill: bool = False,
                 show_abs_error_boxplot: bool = False,
                 abs_error_quantiles_fill: tuple[float, float] | None = None,
                 plot_kwargs: dict[str, Any] | None = None,
                 std_fill_kwargs: dict[str, Any] | None = None,
                 boxplot_kwargs: dict[str, Any] | None = None,
                 quantile_fill_kwargs: dict[str, Any] | None = None
                 ) -> plt.Axes:
        """
        Plots Normalized Mean Absolute Error (NMAE) against the levels of 'variable_col'.
        NMAE is the MAE normalized by the range of target values (or by the mean of target values
        if the range is zero and the mean is non-zero).

        Args:
            ax (Optional[plt.Axes]): Matplotlib Axes object to plot on. If None, creates a new one.
            show_std_fill (bool): If True, shows NMAE +/- std(absolute errors) as a filled area.
                                  'nmae' and 'mae_std' columns must be in per_level_stats_df.
            show_abs_error_boxplot (bool): If True, displays boxplots of normalized absolute errors for each level.
                                           This will use the raw 'preds' and 'targets' from the input df.
            abs_error_quantiles_fill (Optional[Tuple[float, float]]): Tuple (q_low, q_high), e.g., (0.25, 0.75).
                                                                   If provided, fills area between these quantiles
                                                                   of normalized absolute errors.
            plot_kwargs (Optional[Dict[str, Any]]): Keyword arguments for the main NMAE line plot.
            std_fill_kwargs (Optional[Dict[str, Any]]): Keyword arguments for the std deviation fill_between.
            boxplot_kwargs (Optional[Dict[str, Any]]): Keyword arguments for the seaborn boxplot.
            quantile_fill_kwargs (Optional[Dict[str, Any]]): Keyword arguments for the quantile fill_between.

        Returns:
            plt.Axes: The Axes object with the plot.

        Raises:
            ValueError: If required columns for plotting are missing from per_level_stats_df
                        or if 'abs_error_quantiles_fill' provides invalid quantiles.
        """
        ax = self._prepare_ax(ax)
        if self.per_level_stats_df.empty or self.variable_col not in self.per_level_stats_df.columns:
            ax.text(0.5, 0.5, "No data to plot for NMAE.", ha='center', va='center')
            ax.set_title(f"NMAE vs {self.variable_col}")
            return ax

        # Remove direct Series printing
        # print(self.per_level_stats_df[self.variable_col])

        if 'nmae' not in self.per_level_stats_df.columns:
            raise ValueError("Column 'nmae' not found in per_level_stats_df. Cannot plot NMAE.")

        # x_values are the actual category labels (e.g., [1, 2, ..., 10])
        x_categories = self.per_level_stats_df[self.variable_col].tolist()
        y_nmae = self.per_level_stats_df['nmae'].tolist()

        # positions are 0-indexed for plotting (e.g., [0, 1, ..., 9])
        x_positions = list(range(len(x_categories)))


        _plot_kwargs = {'label': 'NMAE', 'marker': 'o'}
        if plot_kwargs: _plot_kwargs.update(plot_kwargs)
        ax.plot(x_positions, y_nmae, **_plot_kwargs) # Use positions for line plot

        if show_std_fill:
            if 'mae_std' not in self.per_level_stats_df.columns:
                raise ValueError("Column 'mae_std' not found in per_level_stats_df. Cannot show std fill.")
            mae_std_values = self.per_level_stats_df['mae_std'].tolist()

            # For NMAE, we need to normalize the std values too
            # We'll use the same normalization factor for each level
            normalized_std_values = []
            for i, level in enumerate(x_categories):
                level_df = self.df[self.df[self.variable_col] == level]
                if not level_df.empty:
                    targets = pd.to_numeric(level_df['targets'], errors='coerce')
                    target_range = targets.max() - targets.min()
                    if target_range > 0:
                        normalization_factor = target_range
                    elif targets.mean() != 0:
                        normalization_factor = abs(targets.mean())
                    else:
                        normalization_factor = 1.0  # Default if both range and mean are zero

                    normalized_std_values.append(mae_std_values[i] / normalization_factor)
                else:
                    normalized_std_values.append(np.nan)

            _std_fill_kwargs = {'alpha': 0.2, 'label': 'NMAE +/- Std(Norm Abs Errors)'}
            if std_fill_kwargs: _std_fill_kwargs.update(std_fill_kwargs)
            ax.fill_between(x_positions,
                            [m - s for m, s in zip(y_nmae, normalized_std_values, strict=False)],
                            [m + s for m, s in zip(y_nmae, normalized_std_values, strict=False)],
                            **_std_fill_kwargs) # Use positions

        temp_df = self.df.copy()
        preds_numeric = pd.to_numeric(temp_df['preds'], errors='coerce')
        targets_numeric = pd.to_numeric(temp_df['targets'], errors='coerce')

        valid_idx = preds_numeric.notna() & targets_numeric.notna()
        temp_df['abs_error'] = np.nan
        if valid_idx.any():
            temp_df.loc[valid_idx, 'abs_error'] = (preds_numeric[valid_idx] - targets_numeric[valid_idx]).abs()

            # Calculate normalized absolute error for each row
            temp_df['norm_abs_error'] = np.nan

            # Group by variable_col to calculate normalization factors
            for _, group in temp_df[valid_idx].groupby(self.variable_col):
                targets_group = pd.to_numeric(group['targets'], errors='coerce')
                target_range = targets_group.max() - targets_group.min()

                if target_range > 0:
                    normalization_factor = target_range
                elif targets_group.mean() != 0:
                    normalization_factor = abs(targets_group.mean())
                else:
                    normalization_factor = 1.0  # Default if both range and mean are zero

                # Apply normalization
                level_indices = group.index
                temp_df.loc[level_indices, 'norm_abs_error'] = temp_df.loc[level_indices, 'abs_error'] / normalization_factor

        plot_df_abs_error = temp_df.dropna(subset=['norm_abs_error', self.variable_col])

        if show_abs_error_boxplot:
            if not plot_df_abs_error.empty:
                _boxplot_kwargs = {
                    'medianprops': {'color': 'black', 'linewidth': 1.5},
                    'boxprops': {'facecolor': 'none', 'edgecolor': 'black'}
                }
                if boxplot_kwargs: _boxplot_kwargs.update(boxplot_kwargs)
                try:
                    # Pass original category values to 'order'. Seaborn maps these to positions 0..N-1.

                    sns.boxplot(x=self.variable_col, y='norm_abs_error', data=plot_df_abs_error, ax=ax, order=x_categories, **_boxplot_kwargs)
                except Exception as e:
                    print(f"Could not generate boxplot: {e}")
            else:
                print("Not enough data for normalized absolute error boxplot after filtering NaNs.")

        if abs_error_quantiles_fill:
            if not (isinstance(abs_error_quantiles_fill, tuple) and len(abs_error_quantiles_fill) == 2 and
                    0 <= abs_error_quantiles_fill[0] < abs_error_quantiles_fill[1] <= 1):
                raise ValueError("abs_error_quantiles_fill must be a tuple of two floats (q_low, q_high) between 0 and 1.")

            if not plot_df_abs_error.empty:
                q_low, q_high = abs_error_quantiles_fill
                grouped_abs_errors = plot_df_abs_error.groupby(self.variable_col)['norm_abs_error']
                try:
                    quantiles = grouped_abs_errors.quantile([q_low, q_high])
                    if not quantiles.empty:
                        quantiles_df = quantiles.unstack()
                        # Reindex quantiles_df to align with x_categories, then use x_positions for plotting
                        if not quantiles_df.empty and x_categories:
                            # Ensure quantiles_df.index matches the type of x_categories for reindexing
                            idx_type = pd.Series(x_categories).dtype
                            try:
                                quantiles_df.index = quantiles_df.index.astype(idx_type)
                            except Exception as e_conv:
                                print(f"Warning: could not convert quantiles_df.index to type {idx_type}: {e_conv}")

                            quantiles_df = quantiles_df.reindex(x_categories)


                            _quantile_fill_kwargs = {'alpha': 0.3, 'label': f'Norm Abs Error {q_low*100:.0f}-{q_high*100:.0f}th pct'}
                            if quantile_fill_kwargs: _quantile_fill_kwargs.update(quantile_fill_kwargs)

                            valid_quantile_idx = quantiles_df[q_low].notna() & quantiles_df[q_high].notna()
                            if valid_quantile_idx.any():
                                # Use x_positions for the fill_between x-coordinates
                                ax.fill_between([x_positions[i] for i, valid in enumerate(valid_quantile_idx) if valid],
                                                quantiles_df[q_low][valid_quantile_idx],
                                                quantiles_df[q_high][valid_quantile_idx],
                                                **_quantile_fill_kwargs)
                            else:
                                print("No valid quantile data to plot after alignment/filtering.")
                        else:
                             print("Quantile data frame is empty or x_categories are empty, cannot plot quantile fill.")
                    else:
                        print("No quantile data computed.")
                except Exception as e:
                    print(f"Could not generate quantile fill for normalized absolute errors: {e}")
            else:
                print("Not enough data for normalized absolute error quantile fill after filtering NaNs.")

        ax.set_xlabel(self.variable_col)
        ax.set_ylabel("NMAE")
        ax.set_title(f"NMAE vs {self.variable_col}")

        # Set xticks and labels
        if x_positions:
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_categories)
            ax.set_xlim(x_positions[0] - 0.5, x_positions[-1] + 0.5)

            # Handle x-tick display for plots with many ticks
            if len(x_positions) > 10:
                # Get current tick labels
                x_tick_labels = [label.get_text() for label in ax.get_xticklabels()]

                # Create a mask where we show all ticks below index 10, then only even-indexed ticks after that
                mask = [(i < 10 or (i >= 10 and (i % 2 == 1))) for i in range(len(x_positions))]

                # Apply the mask to create new labels
                new_labels = [label if mask[i] else '' for i, label in enumerate(x_tick_labels)]

                # Set the new labels
                ax.set_xticklabels(new_labels)

        # After all plotting is done, add this line at the end before returning:
        self._set_plot_fonts(ax, title=f"NMAE vs {self.variable_col}")

        if _plot_kwargs.get('label') or (show_std_fill and _std_fill_kwargs.get('label')) or \
           (abs_error_quantiles_fill and not plot_df_abs_error.empty and _quantile_fill_kwargs.get('label')):
            ax.legend()
            self._set_plot_fonts(ax)
        return ax

    def plot_accuracy(self,
                      ax: plt.Axes | None = None,
                      show_std_fill: bool = False,
                      plot_kwargs: dict[str, Any] | None = None,
                      std_fill_kwargs: dict[str, Any] | None = None
                      ) -> plt.Axes:
        """
        Plots mean accuracy against the levels of 'variable_col'.

        Args:
            ax (Optional[plt.Axes]): Matplotlib Axes object to plot on. If None, creates a new one.
            show_std_fill (bool): If True, shows Mean Accuracy +/- Std(Accuracy Indicator) as a filled area.
                                  'mean_accuracy' and 'std_accuracy' columns must be in per_level_stats_df.
            plot_kwargs (Optional[Dict[str, Any]]): Keyword arguments for the main accuracy line plot.
            std_fill_kwargs (Optional[Dict[str, Any]]): Keyword arguments for the std deviation fill_between.

        Returns:
            plt.Axes: The Axes object with the plot.

        Raises:
            ValueError: If required columns 'mean_accuracy' (or 'std_accuracy' if show_std_fill is True)
                        are missing from per_level_stats_df.
        """
        ax = self._prepare_ax(ax)
        if self.per_level_stats_df.empty or self.variable_col not in self.per_level_stats_df.columns:
            ax.text(0.5, 0.5, "No data to plot for Accuracy.", ha='center', va='center')
            ax.set_title(f"Accuracy vs {self.variable_col}")
            return ax

        if 'mean_accuracy' not in self.per_level_stats_df.columns:
            raise ValueError("Column 'mean_accuracy' not found in per_level_stats_df. Cannot plot accuracy.")

        x_categories = self.per_level_stats_df[self.variable_col].tolist()
        y_accuracy = self.per_level_stats_df['mean_accuracy'].tolist()
        x_positions = list(range(len(x_categories)))

        _plot_kwargs = {'label': 'Mean Accuracy', 'marker': 'o'}
        if plot_kwargs: _plot_kwargs.update(plot_kwargs)
        ax.plot(x_positions, y_accuracy, **_plot_kwargs) # Use positions

        if show_std_fill:
            if 'std_accuracy' not in self.per_level_stats_df.columns:
                raise ValueError("Column 'std_accuracy' not found in per_level_stats_df. Cannot show std fill for accuracy.")
            std_accuracy_values = self.per_level_stats_df['std_accuracy'].tolist()
            _std_fill_kwargs = {'alpha': 0.2, 'label': 'Mean Acc +/- Std(Acc)'}
            if std_fill_kwargs: _std_fill_kwargs.update(std_fill_kwargs)
            ax.fill_between(x_positions,
                            [y - s for y, s in zip(y_accuracy, std_accuracy_values, strict=False)],
                            [y + s for y, s in zip(y_accuracy, std_accuracy_values, strict=False)],
                            **_std_fill_kwargs) # Use positions

        ax.set_xlabel(self.variable_col)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy vs {self.variable_col}")

        if x_positions:
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_categories)
            ax.set_xlim(x_positions[0] - 0.5, x_positions[-1] + 0.5)

            # Handle x-tick display for plots with many ticks
            if len(x_positions) > 10:
                # Get current tick labels
                x_tick_labels = [label.get_text() for label in ax.get_xticklabels()]

                # Create a mask where we show all ticks below index 10, then only even-indexed ticks after that
                mask = [(i < 10 or (i >= 10 and (i % 2 == 1))) for i in range(len(x_positions))]

                # Apply the mask to create new labels
                new_labels = [label if mask[i] else '' for i, label in enumerate(x_tick_labels)]

                # Set the new labels
                ax.set_xticklabels(new_labels)

        # After all plotting is done, add this line at the end before returning:
        self._set_plot_fonts(ax, title=f"Accuracy vs {self.variable_col}")

        if _plot_kwargs.get('label') or (show_std_fill and _std_fill_kwargs.get('label')):
            ax.legend()
            self._set_plot_fonts(ax)
        return ax

    def plot_mean_pred_vs_target(self,
                                 ax: plt.Axes | None = None,
                                 show_pred_std_fill: bool = False,
                                 show_pred_boxplot: bool = False,
                                 pred_quantiles_fill: tuple[float, float] | None = None,
                                 pred_plot_kwargs: dict[str, Any] | None = None,
                                 target_plot_kwargs: dict[str, Any] | None = None,
                                 pred_std_fill_kwargs: dict[str, Any] | None = None,
                                 pred_boxplot_kwargs: dict[str, Any] | None = None,
                                 pred_quantile_fill_kwargs: dict[str, Any] | None = None
                                 ) -> plt.Axes:
        """
        Plots mean predictions and mean targets against the levels of 'variable_col'.

        Args:
            ax (Optional[plt.Axes]): Matplotlib Axes object to plot on. If None, creates a new one.
            show_pred_std_fill (bool): If True, shows Mean Pred +/- Std(Preds) as a filled area.
                                       'pred_mean' and 'pred_std' must be in per_level_stats_df.
            show_pred_boxplot (bool): If True, displays boxplots of predictions for each level.
                                      Uses raw 'preds' from the input df.
            pred_quantiles_fill (Optional[Tuple[float, float]]): Tuple (q_low, q_high) for predictions.
                                                               If provided, fills area between these
                                                               quantiles of predictions.
            pred_plot_kwargs (Optional[Dict[str, Any]]): Keyword arguments for the mean predictions line plot.
            target_plot_kwargs (Optional[Dict[str, Any]]): Keyword arguments for the mean targets line plot.
            pred_std_fill_kwargs (Optional[Dict[str, Any]]): KWargs for prediction std deviation fill_between.
            pred_boxplot_kwargs (Optional[Dict[str, Any]]): KWargs for prediction seaborn boxplot.
            pred_quantile_fill_kwargs (Optional[Dict[str, Any]]): KWargs for prediction quantile fill_between.

        Returns:
            plt.Axes: The Axes object with the plot.

        Raises:
            ValueError: If required columns are missing for plotting or invalid quantiles are given.
        """
        ax = self._prepare_ax(ax)
        if self.per_level_stats_df.empty or self.variable_col not in self.per_level_stats_df.columns:
            ax.text(0.5, 0.5, "No data to plot for Predictions/Targets.", ha='center', va='center')
            ax.set_title(f"Mean Prediction vs Target vs {self.variable_col}")
            return ax

        # Remove direct Series printing
        # print(self.per_level_stats_df[self.variable_col])

        if 'pred_mean' not in self.per_level_stats_df.columns or \
           'target_mean' not in self.per_level_stats_df.columns:
            raise ValueError("Columns 'pred_mean' and 'target_mean' must be in per_level_stats_df.")

        x_categories = self.per_level_stats_df[self.variable_col].tolist()
        y_pred_mean = self.per_level_stats_df['pred_mean'].tolist()
        y_target_mean = self.per_level_stats_df['target_mean'].tolist()

        x_positions = list(range(len(x_categories)))

        _pred_plot_kwargs = {'label': 'Mean Prediction', 'marker': 'o'}
        if pred_plot_kwargs: _pred_plot_kwargs.update(pred_plot_kwargs)
        ax.plot(x_positions, y_pred_mean, **_pred_plot_kwargs) # Use positions

        _target_plot_kwargs = {'label': 'Mean Target', 'marker': 'x', 'linestyle': '--'}
        if target_plot_kwargs: _target_plot_kwargs.update(target_plot_kwargs)
        ax.plot(x_positions, y_target_mean, **_target_plot_kwargs) # Use positions

        if show_pred_std_fill:
            if 'pred_std' not in self.per_level_stats_df.columns:
                raise ValueError("Column 'pred_std' not found. Cannot show prediction std fill.")
            pred_std_values = self.per_level_stats_df['pred_std'].tolist()
            _pred_std_fill_kwargs = {'alpha': 0.2, 'label': 'Mean Pred +/- Std(Preds)'}
            if pred_std_fill_kwargs: _pred_std_fill_kwargs.update(pred_std_fill_kwargs)
            ax.fill_between(x_positions,
                            [m - s for m, s in zip(y_pred_mean, pred_std_values, strict=False)],
                            [m + s for m, s in zip(y_pred_mean, pred_std_values, strict=False)],
                            **_pred_std_fill_kwargs) # Use positions

        temp_df_preds = self.df.copy()
        temp_df_preds['preds'] = pd.to_numeric(temp_df_preds['preds'], errors='coerce')
        plot_df_preds = temp_df_preds.dropna(subset=['preds', self.variable_col])

        if show_pred_boxplot:
            if not plot_df_preds.empty:
                _pred_boxplot_kwargs = {
                    'medianprops': {'color': 'black', 'linewidth': 1.5},
                    'boxprops': {'facecolor': 'none', 'edgecolor': 'black'}
                }
                if pred_boxplot_kwargs: _pred_boxplot_kwargs.update(pred_boxplot_kwargs)
                try:

                    sns.boxplot(x=self.variable_col, y='preds', data=plot_df_preds, ax=ax, order=x_categories, **_pred_boxplot_kwargs)
                except Exception as e:
                    print(f"Could not generate prediction boxplot: {e}")
            else:
                print("Not enough data for prediction boxplot after filtering NaNs.")

        if pred_quantiles_fill:
            if not (isinstance(pred_quantiles_fill, tuple) and len(pred_quantiles_fill) == 2 and
                    0 <= pred_quantiles_fill[0] < pred_quantiles_fill[1] <= 1):
                raise ValueError("pred_quantiles_fill must be a tuple (q_low, q_high) between 0 and 1.")

            if not plot_df_preds.empty:
                q_low, q_high = pred_quantiles_fill
                grouped_preds = plot_df_preds.groupby(self.variable_col)['preds']
                try:
                    quantiles = grouped_preds.quantile([q_low, q_high])
                    # Remove debug print
                    # print(plot_df_preds[[self.variable_col, 'preds']].head())
                    if not quantiles.empty:
                        quantiles_df = quantiles.unstack()
                        if not quantiles_df.empty and x_categories:
                            idx_type = pd.Series(x_categories).dtype
                            try:
                                quantiles_df.index = quantiles_df.index.astype(idx_type)
                            except Exception as e_conv:
                                print(f"Warning: could not convert prediction quantiles_df.index to type {idx_type}: {e_conv}")

                            quantiles_df = quantiles_df.reindex(x_categories)


                            _pred_quantile_fill_kwargs = {'alpha': 0.3, 'label': f'Preds {q_low*100:.0f}-{q_high*100:.0f}th pct'}
                            if pred_quantile_fill_kwargs: _pred_quantile_fill_kwargs.update(pred_quantile_fill_kwargs)

                            valid_quantile_idx = quantiles_df[q_low].notna() & quantiles_df[q_high].notna()
                            if valid_quantile_idx.any():
                                ax.fill_between([x_positions[i] for i, valid in enumerate(valid_quantile_idx) if valid],
                                                quantiles_df[q_low][valid_quantile_idx],
                                                quantiles_df[q_high][valid_quantile_idx],
                                                **_pred_quantile_fill_kwargs)
                            else:
                                print("No valid prediction quantile data to plot after alignment/filtering.")
                        else:
                            print("Prediction quantile data frame is empty or x_categories are empty, cannot plot quantile fill.")
                    else:
                        print("No prediction quantile data computed.")
                except Exception as e:
                     print(f"Could not generate quantile fill for predictions: {e}")
            else:
                print("Not enough data for prediction quantile fill after filtering NaNs.")

        ax.set_xlabel(self.variable_col)
        ax.set_ylabel("Value")
        ax.set_title(f"Mean Prediction vs Target vs {self.variable_col}")

        if x_positions:
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_categories)
            ax.set_xlim(x_positions[0] - 0.5, x_positions[-1] + 0.5)

            # Handle x-tick display for plots with many ticks
            if len(x_positions) > 10:
                # Get current tick labels
                x_tick_labels = [label.get_text() for label in ax.get_xticklabels()]

                # Create a mask where we show all ticks below index 10, then only even-indexed ticks after that
                mask = [(i < 10 or (i >= 10 and (i % 2 == 1))) for i in range(len(x_positions))]

                # Apply the mask to create new labels
                new_labels = [label if mask[i] else '' for i, label in enumerate(x_tick_labels)]

                # Set the new labels
                ax.set_xticklabels(new_labels)

        # After all plotting is done, add this line at the end before returning:
        self._set_plot_fonts(ax, title=f"Mean Prediction vs Target vs {self.variable_col}")

        handles, labels = ax.get_legend_handles_labels()
        # Remove duplicate labels if any, preserving order
        by_label = dict(zip(labels, handles, strict=False))
        if by_label: # Only create legend if there are items
            ax.legend(by_label.values(), by_label.keys())
            self._set_plot_fonts(ax)

        return ax

    def plot_prediction_error_distribution(self,
                                           ax: plt.Axes | None = None,
                                           plot_type: str = 'hist',
                                           drop_na_errors: bool = True,
                                           hist_kwargs: dict[str, Any] | None = None,
                                           kde_kwargs: dict[str, Any] | None = None
                                           ) -> plt.Axes:
        """
        Plots the distribution of prediction errors (preds - targets).

        Args:
            ax (Optional[plt.Axes]): Matplotlib Axes object to plot on. If None, creates a new one.
            plot_type (str): Type of plot. Either 'hist' for histogram or 'kde' for Kernel Density Estimate.
                             Defaults to 'hist'.
            drop_na_errors (bool): If True, NaN values in prediction errors will be dropped before plotting.
                                   Defaults to True.
            hist_kwargs (Optional[Dict[str, Any]]): Keyword arguments for sns.histplot.
            kde_kwargs (Optional[Dict[str, Any]]): Keyword arguments for sns.kdeplot.

        Returns:
            plt.Axes: The Axes object with the plot.

        Raises:
            ValueError: If plot_type is not 'hist' or 'kde', or if 'preds' or 'targets' columns are missing.
        """
        ax = self._prepare_ax(ax)

        if 'preds' not in self.df.columns or 'targets' not in self.df.columns:
            raise ValueError("DataFrame must contain 'preds' and 'targets' columns.")

        # Ensure preds and targets are numeric for error calculation
        preds_numeric = pd.to_numeric(self.df['preds'], errors='coerce')
        targets_numeric = pd.to_numeric(self.df['targets'], errors='coerce')

        errors = preds_numeric - targets_numeric

        if drop_na_errors:
            errors = errors.dropna()

        if errors.empty:
            ax.text(0.5, 0.5, "No valid prediction error data to plot.", ha='center', va='center')
            ax.set_title("Prediction Error Distribution")
            return ax

        if plot_type == 'hist':
            _hist_kwargs = {'kde': False} # Default: no KDE overlay on hist unless specified
            if hist_kwargs: _hist_kwargs.update(hist_kwargs)
            sns.histplot(errors, ax=ax, **_hist_kwargs)
            ax.set_ylabel("Frequency")
        elif plot_type == 'kde':
            _kde_kwargs = {}
            if kde_kwargs: _kde_kwargs.update(kde_kwargs)
            sns.kdeplot(errors, ax=ax, **_kde_kwargs)
            ax.set_ylabel("Density")
        else:
            raise ValueError("plot_type must be 'hist' or 'kde'.")

        ax.set_xlabel("Prediction Error (preds - targets)")
        ax.set_title("Distribution of Prediction Errors")

        # After all plotting is done, add this line at the end before returning:
        self._set_plot_fonts(ax, title="Distribution of Prediction Errors")
        return ax

    def plot_confusion_matrix(self,
                                ax: plt.Axes | None = None,
                                labels: list[Any] | None = None,
                                normalize: str | None = None,
                                cmap: str = "Blues",
                                drop_na: bool = True,
                                heatmap_kwargs: dict[str, Any] | None = None
                                ) -> plt.Axes:
        """
        Plots a confusion matrix for 'targets' and 'preds'.

        Note: This method is most suitable for classification tasks or when 'targets'
        and 'preds' represent discrete categories. If they are continuous, consider
        binning them before using this method or interpreting the output carefully.

        Args:
            ax (Optional[plt.Axes]): Matplotlib Axes object to plot on. If None, creates a new one.
            labels (Optional[List[Any]]): Ordered list of labels to index the matrix.
                                          If None, it's inferred from unique sorted values of
                                          targets and predictions present in the data.
            normalize (Optional[str]): Normalizes confusion matrix over the true (rows),
                                     predicted (columns) conditions or all entries.
                                     Accepts 'true', 'pred', 'all'. Defaults to None (no normalization).
            cmap (str): Colormap for the heatmap. Defaults to "Blues".
            drop_na (bool): If True, rows with NaN in 'targets' or 'preds' are dropped before
                            computing the matrix. Defaults to True.
            heatmap_kwargs (Optional[Dict[str, Any]]): Keyword arguments for sns.heatmap.

        Returns:
            plt.Axes: The Axes object with the plot.

        Raises:
            ValueError: If 'normalize' is not one of 'true', 'pred', 'all', or None, or if essential columns are missing.
        """
        ax = self._prepare_ax(ax)

        if 'preds' not in self.df.columns or 'targets' not in self.df.columns:
            raise ValueError("DataFrame must contain 'preds' and 'targets' columns.")

        # Prepare data
        plot_df = self.df[['targets', 'preds']].copy()
        if drop_na:
            plot_df.dropna(subset=['targets', 'preds'], inplace=True)

        if plot_df.empty:
            ax.text(0.5, 0.5, "No valid data for confusion matrix after dropping NaNs.", ha='center', va='center')
            ax.set_title("Confusion Matrix")
            return ax

        y_true = plot_df['targets']
        y_pred = plot_df['preds']

        # Determine row and column labels separately
        if labels is None:
            # For target rows - use only unique target values
            row_labels = sorted(set(y_true.unique()))
            # For prediction columns - use only unique prediction values
            col_labels = sorted(set(y_pred.unique()))
        else:
            # If labels are provided, use the provided labels
            # We use unique targets for rows and the provided labels for columns
            row_labels = sorted(set(y_true.unique()))
            col_labels = labels

        # Generate confusion matrix with separate row and column labels
        # Create an empty matrix with dimensions (rows = target classes, cols = prediction classes)
        cm = np.zeros((len(row_labels), len(col_labels)), dtype=int)

        # Fill the confusion matrix manually
        for i, true_label in enumerate(row_labels):
            for j, pred_label in enumerate(col_labels):
                cm[i, j] = ((y_true == true_label) & (y_pred == pred_label)).sum()

        if normalize:
            if normalize == 'true':
                # Avoid division by zero by adding a small constant
                row_sums = cm.sum(axis=1)[:, np.newaxis]
                cm = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums!=0)
                fmt = '.2f'
            elif normalize == 'pred':
                col_sums = cm.sum(axis=0)[np.newaxis, :]
                cm = np.divide(cm.astype('float'), col_sums, out=np.zeros_like(cm, dtype=float), where=col_sums!=0)
                fmt = '.2f'
            elif normalize == 'all':
                total = cm.sum()
                cm = cm.astype('float') / total if total > 0 else cm.astype('float')
                fmt = '.2f'
            else:
                raise ValueError("normalize must be one of 'true', 'pred', 'all', or None.")
        else:
            fmt = 'd' # Integer format for raw counts

        # Set default heatmap kwargs
        _heatmap_kwargs = {
            'annot': True,
            'fmt': fmt,
            'cmap': cmap,
            # Force y-label rotation directly in the heatmap parameters
            'yticklabels': True  # Keep this True so seaborn uses our labels
        }
        if heatmap_kwargs:
            # If user provided annot_kws, update our default rather than overwrite
            if 'annot_kws' in heatmap_kwargs:
                _heatmap_kwargs['annot_kws'].update(heatmap_kwargs.pop('annot_kws'))
            _heatmap_kwargs.update(heatmap_kwargs)

        # Create the heatmap with our manually constructed matrix
        _ = sns.heatmap(cm, ax=ax, **_heatmap_kwargs)

        ax.set_title(f"Confusion Matrix {' (Normalized: ' + normalize + ')' if normalize else ''}")
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')

        # Set tick positions and labels
        ax.set_xticks(np.arange(len(col_labels)) + 0.5)  # Add 0.5 to center ticks in cells
        ax.set_yticks(np.arange(len(row_labels)) + 0.5)

        # Create labels with every other value visible
        x_mask = np.arange(len(col_labels)) % 2 == 0  # True for every other position (0, 2, 4, etc.)
        y_mask = np.arange(len(row_labels)) % 2 == 0  # True for every other position (0, 2, 4, etc.)

        x_labels = [str(label) if m else '' for label, m in zip(col_labels, x_mask, strict=False)]
        y_labels = [str(label) if m else '' for label, m in zip(row_labels, y_mask, strict=False)]

        # Set the x-labels normally
        ax.set_xticklabels(x_labels)

        # Make y-labels vertical by forcing rotation to 0 (vertical text reads top to bottom)
        ax.set_yticklabels(y_labels, rotation=0, va='center')

        # Apply consistent font sizes to all elements
        self._set_plot_fonts(ax, title=f"Confusion Matrix{' (Normalized: ' + normalize + ')' if normalize else ''}")

        # Ensure colorbar text has the same font size as tick labels
        cbar = ax.collections[0].colorbar
        if cbar:
            cbar.ax.tick_params(labelsize=17)  # Match the tick_fontsize from _set_plot_fonts

        return ax

    def save_figure(self, fig: plt.Figure, filename_stem: str, file_format: str = "pdf") -> None:
        """
        Saves the given matplotlib Figure to a file in the plotter's output directory.

        Args:
            fig (plt.Figure): The matplotlib Figure object to save.
            filename_stem (str): The base name for the file (without extension).
            file_format (str): The format to save the file in (e.g., "pdf", "png"). Defaults to "pdf".

        Raises:
            TypeError: If fig is not a matplotlib.figure.Figure instance.
            ValueError: If filename_stem is empty.
        """
        if not self.output_dir:
            print("Warning: Output directory not set in MetricsPlotter instance. Cannot save figure.")
            return

        if not isinstance(fig, plt.Figure):
            raise TypeError("Input 'fig' must be a matplotlib.figure.Figure instance.")

        if not filename_stem:
            raise ValueError("filename_stem cannot be empty.")

        output_path = os.path.join(self.output_dir, f"{filename_stem}.{file_format}")

        try:
            fig.savefig(output_path, format=file_format, bbox_inches='tight')
            print(f"Figure successfully saved to {output_path}")
        except Exception as e:
            print(f"Error saving figure to {output_path}: {e}")

    def run_plotting_suite(self, variables_to_plot_individually: list[str], plot_choices: dict[str, bool],  suite_output_dir: str, save_combined_pdf_per_variable: bool, save_individual_pngs_per_variable: bool, vars_to_combine: list[str] | None = None, advanced_plot_options: dict[str, Any] | None = None) -> None:
        """
        Runs a suite of plots for multiple variables and saves the results.

        Args:
            variables_to_plot_individually (List[str]): List of column names in the original DataFrame
                                                      to be used as the 'variable_col' for individual sets of plots.
            plot_choices (Dict[str, bool]): Dictionary indicating which plots to generate for each variable.
                                          Keys can be 'mae', 'nmae', 'accuracy', 'pred_vs_target',
                                          'error_distribution', 'confusion_matrix', 'summary_grid'.
                                          'summary_grid' is True by default and generates a 2x3 grid summary plot.
            suite_output_dir (str): The master output directory for all plots generated by this suite run.
                                    Subdirectories will be created for each variable.
            save_combined_pdf_per_variable (bool): If True, saves a single PDF file with all plots
            save_individual_pngs_per_variable (bool): If True, saves all plot individually
            vars_to_combine (Optional[List[str]]): Placeholder for future feature to combine variables in plots.
                                                   Currently not implemented.
            advanced_plot_options (Optional[Dict[str, Any]]): Dictionary with advanced plotting options.
                                                             Supports 'cross_variable_plots' and 'conditional_plots'.

                                                             Example structure:
                                                             {
                                                                 "cross_variable_plots": [
                                                                     {
                                                                         "variable1": "var1_name",
                                                                         "variable2": "var2_name",
                                                                         "metrics": ["mae", "count"],
                                                                         "output_dir": "optional_dir",
                                                                         "save_individual_pngs": True,
                                                                         "save_combined_pdf": False
                                                                     }
                                                                 ],
                                                                 "conditional_plots": [
                                                                     {
                                                                         "primary_variable": "primary_var",
                                                                         "conditional_variable": "condition_var",
                                                                         "output_dir": "conditional_dir",
                                                                         "metrics": ["mae", "accuracy"],
                                                                         "save_individual_pngs": True
                                                                     }
                                                                 ]
                                                             }
        """
        if vars_to_combine:
            print("Warning: 'vars_to_combine' feature is not yet implemented in run_plotting_suite.")

        # Default options for plots within the suite
        default_mae_opts = {'show_std_fill': True, 'show_abs_error_boxplot': True, 'boxplot_kwargs': {'showfliers': False}}
        default_nmae_opts = {'show_std_fill': True, 'show_abs_error_boxplot': True, 'boxplot_kwargs': {'showfliers': False}}
        default_accuracy_opts = {'show_std_fill': True}
        default_pred_target_opts = {'show_pred_std_fill': True, 'show_pred_boxplot': True, 'pred_boxplot_kwargs': {'showfliers': False}}
        default_error_dist_opts = {'plot_type': 'hist', 'hist_kwargs': {'kde': True, 'bins': 20}}
        default_cm_opts = {'normalize': None} # Example: normalize by true class by default

        # Ensure 'summary_grid' is present and defaults to True
        if plot_choices is None:
            plot_choices = {}
        if 'summary_grid' not in plot_choices:
            plot_choices['summary_grid'] = True
        if 'key_subplots' not in plot_choices:
            plot_choices['key_subplots'] = True

        # --- Process individual variable plots first ---
        for variable_name in variables_to_plot_individually:
            if variable_name not in self.df_original.columns:
                print(f"Warning: Variable '{variable_name}' not found in DataFrame. Skipping this variable.")
                continue

            variable_specific_output_dir = os.path.join(suite_output_dir, f"variable_{variable_name.replace(' ', '_').replace('/', '_')}")
            # No need to os.makedirs here, _initialize_for_variable and save_figure will handle it if output_dir is set.

            print(f"\n--- Generating plots for variable: {variable_name} ---")
            try:
                # Re-initialize plotter for the current variable and set its specific output directory
                self._initialize_for_variable(variable_name, variable_specific_output_dir)
                print(f"Outputting plots for '{variable_name}' to: {self.output_dir}")
            except ValueError as e:
                print(f"Error initializing plotter for variable '{variable_name}': {e}. Skipping this variable.")
                continue

            # --- 0. Handle Summary Grid Plot ---
            if plot_choices.get('summary_grid', True):
                try:
                    summary_grid_filename = f"summary_grid_{variable_name.replace(' ', '_').replace('/', '_')}"
                    self.plot_summary_grid(save_path=summary_grid_filename, file_format="pdf")
                    print(f"Saved summary grid PDF for '{variable_name}'.")
                except Exception as e:
                    print(f"Error generating summary grid for '{variable_name}': {e}")
            # --- 0b. Handle Key Subplots Plot ---
            if plot_choices.get('key_subplots', True):
                try:
                    key_subplots_filename = f"key_subplots_{variable_name.replace(' ', '_').replace('/', '_')}"
                    self.plot_key_subplots(save_path=key_subplots_filename, file_format="pdf")
                    print(f"Saved key subplots PDF for '{variable_name}'.")
                except Exception as e:
                    print(f"Error generating key subplots for '{variable_name}': {e}")

            # --- 1. Handle Combined PDF per Variable ---
            if save_combined_pdf_per_variable:
                active_plot_functions = []
                if plot_choices.get('mae', False): active_plot_functions.append((self.plot_mae, default_mae_opts, f"MAE vs {variable_name}"))
                if plot_choices.get('nmae', False): active_plot_functions.append((self.plot_nmae, default_nmae_opts, f"NMAE vs {variable_name}"))
                if plot_choices.get('accuracy', False): active_plot_functions.append((self.plot_accuracy, default_accuracy_opts, f"Accuracy vs {variable_name}"))
                if plot_choices.get('pred_vs_target', False): active_plot_functions.append((self.plot_mean_pred_vs_target, default_pred_target_opts, f"Pred/Target vs {variable_name}"))
                if plot_choices.get('error_distribution', False): active_plot_functions.append((self.plot_prediction_error_distribution, default_error_dist_opts, "Prediction Error Distribution"))
                if plot_choices.get('confusion_matrix', False):
                    # Prepare labels for CM
                    unique_labels_cm = sorted(set(self.df['targets'].dropna().unique()).union(set(self.df['preds'].dropna().unique())) )
                    unique_labels_cm = [int(l) if isinstance(l, float) and l.is_integer() else l for l in unique_labels_cm]
                    cm_opts_with_labels = {**default_cm_opts, 'labels': unique_labels_cm}
                    active_plot_functions.append((self.plot_confusion_matrix, cm_opts_with_labels, f"Confusion Matrix (Normalized: {default_cm_opts.get('normalize')})"))

                if active_plot_functions:
                    num_plots = len(active_plot_functions)
                    fig_combined, axs_combined = plt.subplots(num_plots, 1, figsize=(12, 6 * num_plots), squeeze=False)
                    axs_combined_flat = axs_combined.flatten()

                    for i, (plot_func, opts, title) in enumerate(active_plot_functions):
                        try:
                            plot_func(ax=axs_combined_flat[i], **opts)
                            axs_combined_flat[i].set_title(title) # Override default title for context in combined plot
                        except Exception as e_plot:
                            print(f"Error generating subplot for '{title}' (variable: {variable_name}): {e_plot}")
                            axs_combined_flat[i].text(0.5, 0.5, f"Error: {e_plot}", ha='center', va='center', color='red')

                    fig_combined.tight_layout(pad=3.0, h_pad=1.0)
                    combined_pdf_filename_stem = f"combined_plots_{variable_name.replace(' ', '_').replace('/', '_')}"
                    self.save_figure(fig_combined, combined_pdf_filename_stem, file_format="pdf")
                    plt.close(fig_combined)
                    print(f"Saved combined PDF for '{variable_name}'.")

            # --- 2. Handle Individual PDFs per Variable ---
            if save_individual_pngs_per_variable:
                plot_configs = {
                    'mae': (self.plot_mae, default_mae_opts, "individual_mae"),
                    'nmae': (self.plot_nmae, default_nmae_opts, "individual_nmae"),
                    'accuracy': (self.plot_accuracy, default_accuracy_opts, "individual_accuracy"),
                    'pred_vs_target': (self.plot_mean_pred_vs_target, default_pred_target_opts, "individual_pred_target"),
                    'error_distribution': (self.plot_prediction_error_distribution, default_error_dist_opts, "individual_error_dist"),
                    'confusion_matrix': (self.plot_confusion_matrix, default_cm_opts, "individual_confusion_matrix")
                }

                for plot_key, (plot_func, opts, filename_stem_base) in plot_configs.items():
                    if plot_choices.get(plot_key, False):
                        fig_ind, ax_ind = plt.subplots(figsize=(10, 7)) # Slightly larger for individual plots
                        try:
                            current_opts = opts.copy()
                            if plot_key == 'confusion_matrix': # Special handling for CM labels
                                unique_labels_cm_ind = sorted(set(self.df['targets'].dropna().unique()).union(set(self.df['preds'].dropna().unique())) )
                                unique_labels_cm_ind = [int(l) if isinstance(l, float) and l.is_integer() else l for l in unique_labels_cm_ind]
                                current_opts['labels'] = unique_labels_cm_ind

                            plot_func(ax=ax_ind, **current_opts)
                            # Title is usually set by the plot method itself, which is fine for individual plots

                            filename_stem_complete = f"{filename_stem_base}_{variable_name.replace(' ', '_').replace('/', '_')}"
                            self.save_figure(fig_ind, filename_stem_complete, file_format="pdf")
                            print(f"Saved individual PDF for '{plot_key}' (variable: {variable_name})")
                        except Exception as e_plot_ind:
                            print(f"Error generating individual PDF for '{plot_key}' (variable: {variable_name}): {e_plot_ind}")
                        finally:
                            plt.close(fig_ind)

        # --- Process advanced plot options ---
        if advanced_plot_options:
            # --- 3. Handle Cross-Variable Performance Plots ---
            cross_variable_configs = advanced_plot_options.get('cross_variable_plots', [])
            if cross_variable_configs:
                print("\n--- Generating cross-variable performance plots ---")
                for config_idx, cross_config in enumerate(cross_variable_configs):
                    variable1 = cross_config.get('variable1')
                    variable2 = cross_config.get('variable2')
                    metrics = cross_config.get('metrics', ['mae', 'mean_accuracy', 'count'])
                    output_dir = cross_config.get('output_dir')

                    # If no specific output_dir is provided, create one under the suite directory
                    if not output_dir:
                        output_dir = os.path.join(suite_output_dir, f"cross_variable_{variable1}_vs_{variable2}")

                    # Default to True for save_individual_pngs if not specified
                    save_individual_pngs = cross_config.get('save_individual_pngs', True)
                    # Default to False for save_combined_pdf if not specified
                    save_combined_pdf = cross_config.get('save_combined_pdf', False)
                    # Optional heatmap_kwargs
                    heatmap_kwargs = cross_config.get('heatmap_kwargs')

                    if not variable1 or not variable2:
                        print(f"Error in cross-variable config #{config_idx+1}: Missing variable1 or variable2.")
                        continue

                    if variable1 not in self.df_original.columns:
                        print(f"Error in cross-variable config #{config_idx+1}: Variable '{variable1}' not found in DataFrame.")
                        continue

                    if variable2 not in self.df_original.columns:
                        print(f"Error in cross-variable config #{config_idx+1}: Variable '{variable2}' not found in DataFrame.")
                        continue

                    try:
                        print(f"Generating cross-variable plots for '{variable1}' vs '{variable2}'...")
                        self.plot_cross_variable_performance(
                            variable1_name=variable1,
                            variable2_name=variable2,
                            metrics_to_plot=metrics,
                            cross_plot_output_dir=output_dir,
                            heatmap_kwargs=heatmap_kwargs,
                            save_individual_pngs=save_individual_pngs,
                            save_combined_pdf=save_combined_pdf
                        )
                    except Exception as e_cross:
                        print(f"Error generating cross-variable plots for '{variable1}' vs '{variable2}': {e_cross}")

            # --- 4. Handle Conditional Single Variable Performance Plots ---
            conditional_configs = advanced_plot_options.get('conditional_plots', [])
            if conditional_configs:
                print("\n--- Generating conditional single-variable performance plots ---")
                for config_idx, cond_config in enumerate(conditional_configs):
                    primary_var = cond_config.get('primary_variable')
                    conditional_var = cond_config.get('conditional_variable')
                    metrics = cond_config.get('metrics', ['mae', 'accuracy'])
                    output_dir = cond_config.get('output_dir')

                    # If no specific output_dir is provided, create one under the suite directory
                    if not output_dir:
                        output_dir = os.path.join(suite_output_dir, f"conditional_{primary_var}_by_{conditional_var}")

                    # Default to True for save_individual_pngs if not specified
                    save_individual_pngs = cond_config.get('save_individual_pngs', True)

                    if not primary_var or not conditional_var:
                        print(f"Error in conditional config #{config_idx+1}: Missing primary_variable or conditional_variable.")
                        continue

                    if primary_var not in self.df_original.columns:
                        print(f"Error in conditional config #{config_idx+1}: Primary variable '{primary_var}' not found in DataFrame.")
                        continue

                    if conditional_var not in self.df_original.columns:
                        print(f"Error in conditional config #{config_idx+1}: Conditional variable '{conditional_var}' not found in DataFrame.")
                        continue

                    try:
                        print(f"Generating conditional plots for '{primary_var}' conditioned on '{conditional_var}'...")
                        self.plot_conditional_single_variable_performance(
                            primary_variable=primary_var,
                            conditional_variable=conditional_var,
                            conditional_plot_output_dir=output_dir,
                            metrics_to_plot=metrics,
                            save_individual_pngs=save_individual_pngs
                        )
                    except Exception as e_cond:
                        print(f"Error generating conditional plots for '{primary_var}' conditioned on '{conditional_var}': {e_cond}")

        print("\nPlotting suite finished.")

    def plot_cross_variable_performance(self,
                                        variable1_name: str,
                                        variable2_name: str,
                                        metrics_to_plot: list[str],
                                        cross_plot_output_dir: str | None = None,
                                        heatmap_kwargs: dict[str, Any] | None = None,
                                        save_individual_pngs: bool = True,
                                        save_combined_pdf: bool = False) -> None:
        """
        Generates and saves heatmaps of specified metrics for combinations of two variables.

        Args:
            variable1_name (str): Name of the first variable (column in self.df_original).
            variable2_name (str): Name of the second variable (column in self.df_original).
            metrics_to_plot (List[str]): List of metric names (e.g., 'mae', 'mean_error', 'count')
                                         to generate heatmaps for. These should be columns in the
                                         output of MetricsCalculator.calculate_cross_variable_stats.
            cross_plot_output_dir (Optional[str]): Specific directory to save these cross-variable plots.
                                                   If None, uses self.output_dir.
            heatmap_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments for sns.heatmap.
            save_individual_pngs (bool): If True, saves each heatmap as a separate PDF.
            save_combined_pdf (bool): If True, saves all generated heatmaps into a single PDF.
        """
        if variable1_name not in self.df_original.columns or variable2_name not in self.df_original.columns:
            print(f"Error: One or both variables ('{variable1_name}', '{variable2_name}') not in DataFrame. Cannot create cross plots.")
            return

        output_dir = cross_plot_output_dir if cross_plot_output_dir is not None else self.output_dir
        if not output_dir:
            print("Warning: Output directory for cross-variable plots is not set. Plots will not be saved.")
            # Optionally, could decide to just display if in an interactive environment, but for now, focus on saving.
            # return
        else:
            try:
                os.makedirs(output_dir, exist_ok=True)
                print(f"Cross-variable plots will be saved to: {output_dir}")
            except Exception as e:
                print(f"Error creating output directory '{output_dir}' for cross-variable plots: {e}. Plots may not save.")
                # return

        # Use a MetricsCalculator instance with the original full DataFrame
        # No group_by_cols needed for this calculator as calculate_cross_variable_stats handles its own grouping.
        calculator = MetricsCalculator(self.df_original.copy())
        try:
            cross_stats_df = calculator.calculate_cross_variable_stats(variable1_name, variable2_name)
        except ValueError as ve:
            print(f"Error calculating cross-variable stats: {ve}")
            return

        if cross_stats_df.empty:
            print(f"No data available for cross-variable analysis between '{variable1_name}' and '{variable2_name}'.")
            return

        # Store figures if a combined PDF is needed
        figures_for_pdf: list[tuple[plt.Figure, str]] = []

        for metric_name in metrics_to_plot:
            if metric_name not in cross_stats_df.columns:
                print(f"Warning: Metric '{metric_name}' not found in cross_stats_df. Skipping this heatmap.")
                continue

            try:
                pivot_df = cross_stats_df.pivot_table(index=variable1_name, columns=variable2_name, values=metric_name)
            except Exception as e_pivot:
                print(f"Error creating pivot table for metric '{metric_name}' (variables: '{variable1_name}', '{variable2_name}'): {e_pivot}. Skipping heatmap.")
                continue

            if pivot_df.empty:
                print(f"Pivot table for metric '{metric_name}' is empty. Skipping heatmap.")
                continue

            fig, ax = plt.subplots(figsize=(max(8, len(pivot_df.columns) * 0.8), max(6, len(pivot_df.index) * 0.6))) # Adjust size

            # Default heatmap kwargs. User can override including vmin, vmax, cmap.
            _h_kwargs = {'annot': True, 'fmt': ".2f", 'cmap': "YlGnBu"}
            if metric_name == 'count': # Counts are usually integers
                _h_kwargs['fmt'] = "d"
            if metric_name == 'mean_accuracy':
                 _h_kwargs['vmin'] = 0.0 # Accuracy typically between 0 and 1
                 _h_kwargs['vmax'] = 1.0

            # Allow user to override defaults or add new ones like vmin, vmax
            if heatmap_kwargs: _h_kwargs.update(heatmap_kwargs)

            try:
                sns.heatmap(pivot_df, ax=ax, **_h_kwargs)
                ax.set_title(f"{metric_name} by {variable1_name} and {variable2_name}")
                ax.set_xlabel(str(variable2_name))
                ax.set_ylabel(str(variable1_name))
                fig.tight_layout()
            except Exception as e_heatmap:
                print(f"Error during heatmap generation for metric '{metric_name}': {e_heatmap}")
                plt.close(fig)
                continue

            filename_stem = f"crossplot_{variable1_name}_vs_{variable2_name}_{metric_name}"

            if save_individual_pngs and output_dir:
                # For saving individual plots, we need to use the MetricsPlotter's save_figure method which expects output_dir to be self.output_dir
                # So, we temporarily set self.output_dir if a specific cross_plot_output_dir was given
                original_plotter_output_dir = self.output_dir
                self.output_dir = output_dir
                try:
                    self.save_figure(fig, filename_stem, file_format="pdf")
                finally:
                    self.output_dir = original_plotter_output_dir # Restore it

            if save_combined_pdf:
                figures_for_pdf.append((fig, filename_stem)) # Store fig for PDF, close later
            else:
                plt.close(fig) # Close if not needed for PDF

        if save_combined_pdf and figures_for_pdf and output_dir:
            from matplotlib.backends.backend_pdf import PdfPages
            pdf_filename = os.path.join(output_dir, f"crossplots_{variable1_name}_vs_{variable2_name}_summary.pdf")
            try:
                with PdfPages(pdf_filename) as pdf:
                    for fig_to_save, _ in figures_for_pdf:
                        pdf.savefig(fig_to_save, bbox_inches='tight')
                        plt.close(fig_to_save) # Close after saving to PDF
                print(f"Saved combined PDF: {pdf_filename}")
            except Exception as e_pdf:
                print(f"Error saving combined PDF for cross-variable plots: {e_pdf}")
                # Ensure any remaining open figures are closed if PDF saving fails mid-way
                for fig_to_save, _ in figures_for_pdf:
                    if plt.fignum_exists(fig_to_save.number):
                        plt.close(fig_to_save)
        elif figures_for_pdf: # If PDF not saved but figures were collected, close them
             for fig_to_save, _ in figures_for_pdf:
                if plt.fignum_exists(fig_to_save.number):
                    plt.close(fig_to_save)

    def plot_conditional_single_variable_performance(
        self,
        primary_variable: str,
        conditional_variable: str,
        conditional_plot_output_dir: str,
        metrics_to_plot: list[str] | None = None,
        save_individual_pngs: bool = True
    ) -> None:
        """
        Plots MAE and Accuracy for a primary_variable, conditioned on fixed levels of a conditional_variable.

        For the conditional_variable, it selects 2 or 3 representative fixed levels:
        - 1 unique level: use that one.
        - 2 unique levels: use both.
        - 3 unique levels: use all three.
        - >3 unique levels (numeric): use min, median, max of unique sorted levels.
        - >3 unique levels (categorical): use first, middle, last of unique sorted levels.

        Args:
            primary_variable (str): The variable to plot on the x-axis.
            conditional_variable (str): The variable whose levels will be fixed.
            conditional_plot_output_dir (str): Base directory to save these conditional plots.
                                               Subdirectories will be created for each fixed level.
            metrics_to_plot (Optional[List[str]]): List of metrics to plot. Defaults to ['mae', 'accuracy'].
            save_individual_pngs (bool): If True, saves each plot as a PDF.
        """
        if metrics_to_plot is None:
            metrics_to_plot = ['mae', 'accuracy']

        if primary_variable not in self.df_original.columns or conditional_variable not in self.df_original.columns:
            print(f"Error: Primary ('{primary_variable}') or conditional ('{conditional_variable}') variable not in DataFrame. Cannot proceed.")
            return

        if primary_variable == conditional_variable:
            print(f"Error: Primary and conditional variables must be different. Both are '{primary_variable}'.")
            return

        try:
            os.makedirs(conditional_plot_output_dir, exist_ok=True)
        except Exception as e_mkdir:
            print(f"Error creating base directory for conditional plots '{conditional_plot_output_dir}': {e_mkdir}. Plots may not save.")
            # return # Decide if this is a hard stop

        unique_conditional_levels = sorted(self.df_original[conditional_variable].dropna().unique())

        fixed_levels_to_use = []
        if not unique_conditional_levels:
            print(f"No unique, non-NaN levels found for conditional variable '{conditional_variable}'. Cannot generate conditional plots.")
            return

        num_unique = len(unique_conditional_levels)
        if num_unique == 1 or num_unique == 2 or num_unique == 3:
            fixed_levels_to_use = unique_conditional_levels
        else: # > 3 unique levels
            if pd.api.types.is_numeric_dtype(self.df_original[conditional_variable].dropna()):
                # Min, Median, Max of the unique levels themselves
                fixed_levels_to_use = [
                    unique_conditional_levels[0],
                    unique_conditional_levels[num_unique // 2], # Median element of sorted unique levels
                    unique_conditional_levels[-1]
                ]
            else: # Categorical
                fixed_levels_to_use = [
                    unique_conditional_levels[0],
                    unique_conditional_levels[num_unique // 2],
                    unique_conditional_levels[-1]
                ]
        fixed_levels_to_use = sorted(set(fixed_levels_to_use)) # Ensure uniqueness if min/median/max coincide

        print(f"Conditional plots for '{primary_variable}' fixing '{conditional_variable}' at levels: {fixed_levels_to_use}")

        for fixed_level in fixed_levels_to_use:
            filtered_df = self.df_original[self.df_original[conditional_variable] == fixed_level].copy()

            if filtered_df.empty:
                print(f"No data found for '{primary_variable}' when '{conditional_variable}' is {fixed_level}. Skipping.")
                continue
            if filtered_df[primary_variable].nunique() < 1:
                 print(f"Not enough unique values of primary_variable '{primary_variable}' ({filtered_df[primary_variable].nunique()}) when '{conditional_variable}' is {fixed_level}. Skipping plots for this level.")
                 continue

            level_str = str(fixed_level).replace('.', 'p').replace(' ', '_').replace('/', '_')
            level_output_dir = os.path.join(conditional_plot_output_dir, f"{conditional_variable}_fixed_at_{level_str}")

            # Use a temporary plotter instance for this specific filtered data and primary_variable
            try:
                # The output_dir for this temp_plotter is crucial for save_figure
                temp_plotter = MetricsPlotter(df=filtered_df, variable_col=primary_variable, output_dir=level_output_dir)
            except ValueError as ve:
                print(f"Error initializing temp_plotter for {primary_variable} (conditional {conditional_variable}={fixed_level}): {ve}. Skipping this level.")
                continue

            print(f"  Generating plots for '{primary_variable}' with '{conditional_variable}' = {fixed_level} (Output: {level_output_dir})")

            if 'mae' in metrics_to_plot:
                try:
                    fig_mae, ax_mae = plt.subplots(figsize=(10,6))
                    temp_plotter.plot_mae(ax=ax_mae, show_std_fill=True, show_abs_error_boxplot=True, boxplot_kwargs={'showfliers': False})
                    ax_mae.set_title(f"MAE vs {primary_variable}\n(Condition: {conditional_variable} = {fixed_level})")
                    if save_individual_pngs:
                        temp_plotter.save_figure(fig_mae, f"mae_vs_{primary_variable}", file_format="pdf")
                    plt.close(fig_mae)
                except Exception as e_m:
                    print(f"    Error plotting MAE for {primary_variable} ({conditional_variable}={fixed_level}): {e_m}")
                    if plt.fignum_exists(fig_mae.number): plt.close(fig_mae)

            if 'accuracy' in metrics_to_plot:
                try:
                    fig_acc, ax_acc = plt.subplots(figsize=(10,6))
                    temp_plotter.plot_accuracy(ax=ax_acc, show_std_fill=True)
                    ax_acc.set_title(f"Accuracy vs {primary_variable}\n(Condition: {conditional_variable} = {fixed_level})")
                    if save_individual_pngs:
                        temp_plotter.save_figure(fig_acc, f"accuracy_vs_{primary_variable}", file_format="pdf")
                    plt.close(fig_acc)
                except Exception as e_a:
                    print(f"    Error plotting Accuracy for {primary_variable} ({conditional_variable}={fixed_level}): {e_a}")
                    if plt.fignum_exists(fig_acc.number): plt.close(fig_acc)

        print(f"Finished conditional plotting for {primary_variable} based on {conditional_variable}.")

    def plot_summary_grid(
        self,
        save_path: str | None = None,
        file_format: str = "pdf",
        plot_kwargs: dict[str, Any] | None = None
    ) -> plt.Figure:
        """
        Creates a 2x3 grid of subplots summarizing key metrics:
        Row 1: MAE, Mean Predicted vs Target, NMAE
        Row 2: Accuracy, Confusion Matrix, Error Distribution

        Args:
            save_path (Optional[str]): If provided, saves the figure to this path (without extension).
            file_format (str): File format for saving (default: "pdf").
            plot_kwargs (Optional[Dict[str, Any]]): Optional dict for advanced plot customizations.

        Returns:
            plt.Figure: The matplotlib Figure object.
        """
        if plot_kwargs is None:
            plot_kwargs = {}
        fig, axs = plt.subplots(2, 3, figsize=(24, 16))  # Increased figure size for better readability
        fig.subplots_adjust(hspace=0.3, wspace=0.3)  # More space between subplots

        # Row 1
        # (0,0): MAE
        try:
            self.plot_mae(ax=axs[0, 0], show_std_fill=True, show_abs_error_boxplot=True, boxplot_kwargs={'showfliers': False})
            axs[0, 0].set_title(f"MAE vs {self.variable_col}", fontsize=30)
        except Exception as e:
            axs[0, 0].text(0.5, 0.5, f"Error plotting MAE: {e}", ha='center', va='center', color='red', fontsize=20)
            axs[0, 0].set_title("MAE (Error)", fontsize=30)

        # (0,1): Mean Predicted vs Target
        try:
            self.plot_mean_pred_vs_target(ax=axs[0, 1], show_pred_std_fill=True, show_pred_boxplot=True, pred_boxplot_kwargs={'showfliers': False})
            axs[0, 1].set_title(f"Pred/Target vs {self.variable_col}", fontsize=30)
        except Exception as e:
            axs[0, 1].text(0.5, 0.5, f"Error plotting Pred/Target: {e}", ha='center', va='center', color='red', fontsize=20)
            axs[0, 1].set_title("Mean Pred/Target (Error)", fontsize=30)

        # (0,2): NMAE
        try:
            self.plot_nmae(ax=axs[0, 2], show_std_fill=True, show_abs_error_boxplot=True, boxplot_kwargs={'showfliers': False})
            axs[0, 2].set_title(f"NMAE vs {self.variable_col}", fontsize=30)
        except Exception as e:
            axs[0, 2].text(0.5, 0.5, f"Error plotting NMAE: {e}", ha='center', va='center', color='red', fontsize=20)
            axs[0, 2].set_title("NMAE (Error)", fontsize=30)

        # Row 2
        # (1,0): Accuracy
        try:
            self.plot_accuracy(ax=axs[1, 0], show_std_fill=True)
            axs[1, 0].set_title(f"Accuracy vs {self.variable_col}", fontsize=30)
        except Exception as e:
            axs[1, 0].text(0.5, 0.5, f"Error plotting Accuracy: {e}", ha='center', va='center', color='red', fontsize=20)
            axs[1, 0].set_title("Accuracy (Error)", fontsize=30)

        # (1,1): Confusion Matrix
        try:
            # Infer labels for confusion matrix from the data
            unique_labels = sorted(set(self.df['targets'].dropna().unique()).union(set(self.df['preds'].dropna().unique())))
            unique_labels = [int(l) if isinstance(l, float) and l.is_integer() else l for l in unique_labels]

            # Plot the confusion matrix with default parameters
            self.plot_confusion_matrix(ax=axs[1, 1], labels=unique_labels, normalize=None)
            axs[1, 1].set_title("Confusion Matrix", fontsize=30)

            # Manually adjust font sizes after the plot is created
            # This approach avoids the annot_kws parameter issue
            for text in axs[1, 1].texts:
                if text.get_text().strip():  # Only adjust non-empty text
                    text.set_fontsize(15)  # Increased from 8 to 13

        except Exception as e:
            axs[1, 1].text(0.5, 0.5, f"Error plotting Confusion Matrix: {e}", ha='center', va='center', color='red', fontsize=20)
            axs[1, 1].set_title("Confusion Matrix (Error)", fontsize=30)

        # (1,2): Error Distribution
        try:
            self.plot_prediction_error_distribution(ax=axs[1, 2], plot_type='hist', hist_kwargs={'bins': 20, 'kde': True})
            axs[1, 2].set_title("Prediction Error Distribution", fontsize=30)
        except Exception as e:
            axs[1, 2].text(0.5, 0.5, f"Error plotting Error Distribution: {e}", ha='center', va='center', color='red', fontsize=20)
            axs[1, 2].set_title("Error Distribution (Error)", fontsize=30)

        # Apply consistent font sizes to all subplots except confusion matrix
        for i in range(2):
            for j in range(3):
                if not (i == 1 and j == 1):  # Skip the confusion matrix
                    self._set_plot_fonts(axs[i, j])

        fig.tight_layout(pad=4.0, h_pad=2.0, w_pad=2.0)

        if save_path:
            self.save_figure(fig, save_path, file_format=file_format)

        return fig

    def plot_key_subplots(
        self,
        save_path: str | None = None,
        file_format: str = "pdf",
        plot_kwargs: dict[str, Any] | None = None
    ) -> plt.Figure:
        """
        Creates a 2x2 grid of subplots with key metrics:
        (0,0): MAE
        (0,1): Mean Prediction vs Target
        (1,0): Confusion Matrix
        (1,1): Error Distribution

        Args:
            save_path (Optional[str]): If provided, saves the figure to this path (without extension).
            file_format (str): File format for saving (default: "pdf").
            plot_kwargs (Optional[Dict[str, Any]]): Optional dict for advanced plot customizations.

        Returns:
            plt.Figure: The matplotlib Figure object.
        """
        if plot_kwargs is None:
            plot_kwargs = {}
        fig, axs = plt.subplots(2, 2, figsize=(24, 16))  # Increased figure size
        fig.subplots_adjust(hspace=0.3, wspace=0.3)  # More space between subplots

        # Font size settings
        title_fontsize = 30
        axis_fontsize = 20
        tick_fontsize = 20
        legend_fontsize = 25

        # (0,0): MAE
        try:
            self.plot_mae(ax=axs[0, 0], show_std_fill=True, show_abs_error_boxplot=True, boxplot_kwargs={'showfliers': False})
            axs[0, 0].set_title(f"MAE vs {self.variable_col}", fontsize=title_fontsize)
            axs[0, 0].set_xlabel(axs[0, 0].get_xlabel(), fontsize=axis_fontsize)
            axs[0, 0].set_ylabel(axs[0, 0].get_ylabel(), fontsize=axis_fontsize)
            axs[0, 0].tick_params(axis='both', labelsize=tick_fontsize)
            legend = axs[0, 0].get_legend()
            if legend is not None:
                for text in legend.get_texts():
                    text.set_fontsize(legend_fontsize)
        except Exception as e:
            axs[0, 0].text(0.5, 0.5, f"Error plotting MAE: {e}", ha='center', va='center', color='red', fontsize=20)
            axs[0, 0].set_title("MAE (Error)", fontsize=title_fontsize)

        # (0,1): Mean Prediction vs Target
        try:
            self.plot_mean_pred_vs_target(ax=axs[0, 1], show_pred_std_fill=True, show_pred_boxplot=True, pred_boxplot_kwargs={'showfliers': False})
            axs[0, 1].set_title(f"Mean Pred/Target vs {self.variable_col}", fontsize=title_fontsize)
            axs[0, 1].set_xlabel(axs[0, 1].get_xlabel(), fontsize=axis_fontsize)
            axs[0, 1].set_ylabel(axs[0, 1].get_ylabel(), fontsize=axis_fontsize)
            axs[0, 1].tick_params(axis='both', labelsize=tick_fontsize)
            legend = axs[0, 1].get_legend()
            if legend is not None:
                for text in legend.get_texts():
                    text.set_fontsize(legend_fontsize)
        except Exception as e:
            axs[0, 1].text(0.5, 0.5, f"Error plotting Pred/Target: {e}", ha='center', va='center', color='red', fontsize=20)
            axs[0, 1].set_title("Mean Pred/Target (Error)", fontsize=title_fontsize)

        # (1,0): Confusion Matrix
        try:
            unique_labels = sorted(set(self.df['targets'].dropna().unique()).union(set(self.df['preds'].dropna().unique())) )
            unique_labels = [int(l) if isinstance(l, float) and l.is_integer() else l for l in unique_labels]
            self.plot_confusion_matrix(ax=axs[1, 0], labels=unique_labels, normalize=None)
            axs[1, 0].set_title("Confusion Matrix", fontsize=axis_fontsize)
            axs[1, 0].set_xlabel(axs[1, 0].get_xlabel(), fontsize=axis_fontsize)
            axs[1, 0].set_ylabel(axs[1, 0].get_ylabel(), fontsize=axis_fontsize)
            axs[1, 0].tick_params(axis='both', labelsize=tick_fontsize)
            # Only show every other tick label for x and y axes
            xlabels = [label.get_text() for label in axs[1, 0].get_xticklabels()]
            ylabels = [label.get_text() for label in axs[1, 0].get_yticklabels()]
            xlabels = [lbl if i % 2 == 0 else '' for i, lbl in enumerate(xlabels)]
            ylabels = [lbl if i % 2 == 0 else '' for i, lbl in enumerate(ylabels)]
            axs[1, 0].set_xticklabels(xlabels)
            axs[1, 0].set_yticklabels(ylabels)
            # Do not change annotation font size inside the heatmap
            legend = axs[1, 0].get_legend()
            if legend is not None:
                for text in legend.get_texts():
                    text.set_fontsize(legend_fontsize)
        except Exception as e:
            axs[1, 0].text(0.5, 0.5, f"Error plotting Confusion Matrix: {e}", ha='center', va='center', color='red')
            axs[1, 0].set_title("Confusion Matrix (Error)", fontsize=axis_fontsize)

        # (1,1): Error Distribution
        try:
            self.plot_prediction_error_distribution(ax=axs[1, 1], plot_type='hist', hist_kwargs={'bins': 20, 'kde': True})
            axs[1, 1].set_title("Prediction Error Distribution", fontsize=axis_fontsize)
            axs[1, 1].set_xlabel(axs[1, 1].get_xlabel(), fontsize=axis_fontsize)
            axs[1, 1].set_ylabel(axs[1, 1].get_ylabel(), fontsize=axis_fontsize)
            axs[1, 1].tick_params(axis='both', labelsize=tick_fontsize)
            legend = axs[1, 1].get_legend()
            if legend is not None:
                for text in legend.get_texts():
                    text.set_fontsize(legend_fontsize)
        except Exception as e:
            axs[1, 1].text(0.5, 0.5, f"Error plotting Error Distribution: {e}", ha='center', va='center', color='red')
            axs[1, 1].set_title("Error Distribution (Error)", fontsize=axis_fontsize)

        fig.tight_layout(pad=3.0, h_pad=1.0)

        if save_path:
            self.save_figure(fig, save_path, file_format=file_format)

        return fig


if __name__ == '__main__':
    # Define the desired output directory name for plots generated by this script run
    script_output_directory = "chess_gpt-4.1_declarative_debiased_count"


    # Load the diagnostic results CSV
    try:
        df_diagnostic = pd.read_csv("diagnostic_results/chess_gpt-4.1_declarative_debiased/gpt-4.1_count/diagnostic_evaluation_results_gpt-4.1_count_pieces_20250509_222736.csv")
    except FileNotFoundError:
        print("Error: diagnostic_evaluation_results_gemma_20250508_120051.csv not found.")
        print("Please ensure the file is in the 'diagnostic_results' directory relative to where this script is run.")
        exit()
    except Exception as e:
        print(f"Error loading CSV: {e}")
        exit()

    print("\n--- Plotting with Diagnostic Data (diagnostic_evaluation_results_gemma_20250508_120051.csv) ---")
    print(f"Loaded diagnostic data with {len(df_diagnostic)} rows.")
    print(f"DEBUG (__main__): Initial unique values of 'target': {df_diagnostic['target'].unique()}")
    print(f"DEBUG (__main__): Initial unique values of 'pred': {df_diagnostic['pred'].unique()}")

    # The 'target' and 'pred' columns from the CSV will be used.
    # 'number' will be the variable_col.
    # MetricsPlotter's __init__ handles renaming 'pred' to 'preds' and 'target' to 'targets',
    # and converts them to numeric.

    variable_to_plot = 'number' # This is the column we'll use for the x-axis

    if variable_to_plot not in df_diagnostic.columns:
        print(f"Error: Column '{variable_to_plot}' not found in the CSV. Available columns: {df_diagnostic.columns.tolist()}")
        exit()
    print(f"DEBUG (__main__): Unique values of '{variable_to_plot}' in df_diagnostic: {df_diagnostic[variable_to_plot].unique()}")
    if 'target' not in df_diagnostic.columns:
        print(f"Error: Column 'target' not found in the CSV. Available columns: {df_diagnostic.columns.tolist()}")
        exit()
    if 'pred' not in df_diagnostic.columns:
        print(f"Error: Column 'pred' not found in the CSV. Available columns: {df_diagnostic.columns.tolist()}")
        exit()

    try:
        # Pass the output directory to the MetricsPlotter instance
        plotter_diagnostic = MetricsPlotter(df_diagnostic,
                                          variable_col=variable_to_plot,
                                          output_dir=script_output_directory)
    except Exception as e:
        print(f"Error initializing MetricsPlotter: {e}")
        exit()

    print(f"DEBUG (__main__): plotter_diagnostic.per_level_stats_df[{variable_to_plot}] before plotting:")
    print(plotter_diagnostic.per_level_stats_df[variable_to_plot])
    print("DEBUG (__main__): plotter_diagnostic.per_level_stats_df full before plotting:")
    print(plotter_diagnostic.per_level_stats_df.to_string())

    # Create a figure with 5 subplots for the original 3 + 2 new ones
    fig, axs = plt.subplots(5, 1, figsize=(12, 32)) # Increased figure size for 5 plots
    fig.tight_layout(pad=5.0) # Increased padding

    # Plot MAE
    try:
        plotter_diagnostic.plot_mae(
            ax=axs[0],
            show_std_fill=True,
            show_abs_error_boxplot=True,
            abs_error_quantiles_fill=(0.25, 0.75),
            boxplot_kwargs={'showfliers': False}
        )
        axs[0].set_title(f"MAE vs {variable_to_plot} (Gemma Diagnostic Data)")
        # Customize x-ticks if 'number' is numeric and has many unique values for clarity
        # if pd.api.types.is_numeric_dtype(plotter_diagnostic.per_level_stats_df[variable_to_plot]):
        #     unique_x_values = sorted(plotter_diagnostic.per_level_stats_df[variable_to_plot].dropna().unique())
        #     print(f"DEBUG (__main__ - MAE plot): unique_x_values for x-ticks: {unique_x_values}")
        #     if unique_x_values:
        #         axs[0].set_xticks(unique_x_values)
        #         axs[0].set_xlim(min(unique_x_values) - 0.5, max(unique_x_values) + 0.5)

    except Exception as e:
        axs[0].text(0.5, 0.5, f"Error plotting MAE: {e}", ha='center', va='center', color='red')
        axs[0].set_title(f"MAE vs {variable_to_plot} (Plotting Error)")
        print(f"Error during MAE plot: {e}")

    # Plot Accuracy
    try:
        plotter_diagnostic.plot_accuracy(
            ax=axs[1],
            show_std_fill=True
        )
        axs[1].set_title(f"Accuracy vs {variable_to_plot} (Gemma Diagnostic Data)")
        # if pd.api.types.is_numeric_dtype(plotter_diagnostic.per_level_stats_df[variable_to_plot]):
        #     unique_x_values = sorted(plotter_diagnostic.per_level_stats_df[variable_to_plot].dropna().unique())
        #     print(f"DEBUG (__main__ - Accuracy plot): unique_x_values for x-ticks: {unique_x_values}")
        #     if unique_x_values:
        #         axs[1].set_xticks(unique_x_values)
        #         axs[1].set_xlim(min(unique_x_values) - 0.5, max(unique_x_values) + 0.5)

    except Exception as e:
        axs[1].text(0.5, 0.5, f"Error plotting Accuracy: {e}", ha='center', va='center', color='red')
        axs[1].set_title(f"Accuracy vs {variable_to_plot} (Plotting Error)")
        print(f"Error during Accuracy plot: {e}")

    # Plot Mean Prediction vs Target
    try:
        plotter_diagnostic.plot_mean_pred_vs_target(
            ax=axs[2],
            show_pred_std_fill=True,
            show_pred_boxplot=True, # Enable boxplot for predictions
            pred_quantiles_fill=(0.25, 0.75),
            pred_boxplot_kwargs={'showfliers': False}
        )
        axs[2].set_title(f"Mean Pred/Target vs {variable_to_plot} (Gemma Diagnostic Data)")
        # if pd.api.types.is_numeric_dtype(plotter_diagnostic.per_level_stats_df[variable_to_plot]):
        #     unique_x_values = sorted(plotter_diagnostic.per_level_stats_df[variable_to_plot].dropna().unique())
        #     print(f"DEBUG (__main__ - Pred/Target plot): unique_x_values for x-ticks: {unique_x_values}")
        #     if unique_x_values:
        #         axs[2].set_xticks(unique_x_values)
        #         axs[2].set_xlim(min(unique_x_values) - 0.5, max(unique_x_values) + 0.5)

    except Exception as e:
        axs[2].text(0.5, 0.5, f"Error plotting Pred/Target: {e}", ha='center', va='center', color='red')
        axs[2].set_title(f"Mean Pred/Target vs {variable_to_plot} (Plotting Error)")
        print(f"Error during Mean Pred/Target plot: {e}")

    # Plot Prediction Error Distribution (Histogram)
    try:
        plotter_diagnostic.plot_prediction_error_distribution(
            ax=axs[3],
            plot_type='hist',
            hist_kwargs={'bins': 20, 'kde': True} # Example: 20 bins and overlay KDE
        )
        axs[3].set_title("Prediction Error Distribution (Gemma Diagnostic Data)")
    except Exception as e:
        axs[3].text(0.5, 0.5, f"Error plotting Error Distribution: {e}", ha='center', va='center', color='red')
        axs[3].set_title("Prediction Error Distribution (Plotting Error)")
        print(f"Error during Prediction Error Distribution plot: {e}")

    # Plot Confusion Matrix
    try:
        # Infer labels for confusion matrix from the data, as an example
        # This assumes 'target' and 'pred' can be treated as class labels
        unique_labels = sorted(set(df_diagnostic['target'].dropna().unique()).union(set(df_diagnostic['pred'].dropna().unique())))
        # Convert to int if they are like 1.0, 2.0 etc.
        unique_labels = [int(l) if isinstance(l, float) and l.is_integer() else l for l in unique_labels]

        plotter_diagnostic.plot_confusion_matrix(
            ax=axs[4],
            labels=unique_labels, # Pass inferred labels
            normalize='true',      # Example: normalize over true labels
            heatmap_kwargs={'annot_kws':{"size": 8}} # Smaller annotations if many cells
        )
        # Title is set within the method, but we can adjust if needed
        # axs[4].set_title(f"Confusion Matrix (Gemma Diagnostic Data)")
    except Exception as e:
        axs[4].text(0.5, 0.5, f"Error plotting Confusion Matrix: {e}", ha='center', va='center', color='red')
        axs[4].set_title("Confusion Matrix (Plotting Error)")
        print(f"Error during Confusion Matrix plot: {e}")

    # Use the new save_figure method from the plotter instance
    plotter_diagnostic.save_figure(fig, "diagnostic_plots_gemma_combined_extended", file_format="pdf")

    # --- Save Individual Plots as PDF ---
    print("\n--- Saving Individual Plots as PDF ---")

    # Individual MAE Plot
    try:
        fig_mae, ax_mae = plt.subplots(figsize=(10, 6))
        plotter_diagnostic.plot_mae(
            ax=ax_mae,
            show_std_fill=True,
            show_abs_error_boxplot=True,
            abs_error_quantiles_fill=(0.25, 0.75),
            boxplot_kwargs={'showfliers': False}
        )
        ax_mae.set_title(f"MAE vs {variable_to_plot} (Gemma Diagnostic Data - Individual)")
        plotter_diagnostic.save_figure(fig_mae, "diagnostic_mae_plot", file_format="pdf")
        plt.close(fig_mae)
    except Exception as e:
        print(f"Error saving individual MAE plot: {e}")

    # Individual Accuracy Plot
    try:
        fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
        plotter_diagnostic.plot_accuracy(
            ax=ax_acc,
            show_std_fill=True
        )
        ax_acc.set_title(f"Accuracy vs {variable_to_plot} (Gemma Diagnostic Data - Individual)")
        plotter_diagnostic.save_figure(fig_acc, "diagnostic_accuracy_plot", file_format="pdf")
        plt.close(fig_acc)
    except Exception as e:
        print(f"Error saving individual Accuracy plot: {e}")

    # Individual Mean Prediction vs Target Plot
    try:
        fig_pred_target, ax_pred_target = plt.subplots(figsize=(10, 6))
        plotter_diagnostic.plot_mean_pred_vs_target(
            ax=ax_pred_target,
            show_pred_std_fill=True,
            show_pred_boxplot=True,
            pred_quantiles_fill=(0.25, 0.75),
            pred_boxplot_kwargs={'showfliers': False}
        )
        ax_pred_target.set_title(f"Mean Pred/Target vs {variable_to_plot} (Gemma Diagnostic Data - Individual)")
        plotter_diagnostic.save_figure(fig_pred_target, "diagnostic_pred_target_plot", file_format="pdf")
        plt.close(fig_pred_target)
    except Exception as e:
        print(f"Error saving individual Mean Pred/Target plot: {e}")

    # Individual Prediction Error Distribution Plot
    try:
        fig_err_dist, ax_err_dist = plt.subplots(figsize=(10, 6))
        plotter_diagnostic.plot_prediction_error_distribution(
            ax=ax_err_dist,
            plot_type='hist',
            hist_kwargs={'bins': 30, 'edgecolor': 'black', 'kde': True}
        )
        ax_err_dist.set_title("Prediction Error Distribution (Gemma Diagnostic Data - Individual)")
        plotter_diagnostic.save_figure(fig_err_dist, "diagnostic_error_distribution_plot", file_format="pdf")
        plt.close(fig_err_dist)
    except Exception as e:
        print(f"Error saving individual Prediction Error Distribution plot: {e}")

    # Individual Confusion Matrix Plot
    try:
        fig_cm, ax_cm = plt.subplots(figsize=(10, 8)) # Adjusted size for CM
        unique_labels_cm_indiv = sorted(set(df_diagnostic['target'].dropna().unique()).union(set(df_diagnostic['pred'].dropna().unique())) )
        unique_labels_cm_indiv = [int(l) if isinstance(l, float) and l.is_integer() else l for l in unique_labels_cm_indiv]

        plotter_diagnostic.plot_confusion_matrix(
            ax=ax_cm,
            labels=unique_labels_cm_indiv,
            normalize=None, # Show raw counts for this one
            cmap="viridis"
        )
        plotter_diagnostic.save_figure(fig_cm, "diagnostic_confusion_matrix_plot", file_format="pdf")
        plt.close(fig_cm)
    except Exception as e:
        print(f"Error saving individual Confusion Matrix plot: {e}")

    plt.close(fig) # Close the main combined figure
    print("\nDiagnostic plotting script finished.")


    # --- Example usage of the new run_plotting_suite method ---
    print("\n\n--- Running plotting suite --- ")
    if not df_diagnostic.empty:
        # Create a base plotter instance just to call the suite method from
        # The actual variable_col and output_dir for this instance don't matter much for the suite call itself,
        # as the suite will create new instances or re-initialize for each variable it processes.
        # However, it needs some valid variable_col from the df for the initial instantiation.
        initial_var_col_for_suite_runner = 'number' # or any valid column from df_diagnostic
        if initial_var_col_for_suite_runner not in df_diagnostic.columns:
            print(f"Error: Column '{initial_var_col_for_suite_runner}' needed for initial plotter setup not found. Cannot run suite.")
        else:
            suite_plotter = MetricsPlotter(df_diagnostic,
                                         variable_col=initial_var_col_for_suite_runner,
                                         output_dir="suite_general_output_temp") # Temporary output dir

            variables_to_analyze = ['number'] # Could be ['number', 'another_categorical_column'] if available

            # Example: Add another dummy variable column to df_diagnostic for testing the suite with multiple variables
            if 'number' in df_diagnostic.columns:
                if len(df_diagnostic['number'].unique()) > 1:
                    # Create a second variable based on 'number' for demonstration
                    df_diagnostic['number_group'] = pd.qcut(df_diagnostic['number'], q=2, labels=["group1", "group2"], duplicates='drop')
                    if 'number_group' in df_diagnostic.columns and df_diagnostic['number_group'].nunique() > 1:
                        variables_to_analyze.append('number_group')
                    else:
                        print("Could not create a dummy 'number_group' with enough unique values for suite demonstration.")
                else:
                    print("Not enough unique values in 'number' to create 'number_group' for suite demo.")

            plot_selection = {
                'mae': True,
                'nmae': True,
                'accuracy': True,
                'pred_vs_target': True,
                'error_distribution': True,
                'confusion_matrix': True,
                'summary_grid': True,
                'key_subplots': True
            }

            suite_plotter.run_plotting_suite(
                variables_to_plot_individually=variables_to_analyze,
                plot_choices=plot_selection,
                suite_output_dir="suite_plots_output", # Master directory for this suite run
                save_combined_pdf_per_variable=True,
                save_individual_pngs_per_variable=True,
                vars_to_combine=None, # Placeholder for future feature
                advanced_plot_options={
                    "cross_variable_plots": [
                        {
                            "variable1": "number",
                            "variable2": "number_group",
                            "metrics": ["mae", "count"],
                            "output_dir": "cross_variable_plots",
                            "save_individual_pngs": True,
                            "save_combined_pdf": False
                        }
                    ],
                    "conditional_plots": [
                        {
                            "primary_variable": "number",
                            "conditional_variable": "number_group",
                            "output_dir": "conditional_plots",
                            "metrics": ["mae", "accuracy"],
                            "save_individual_pngs": True
                        }
                    ]
                }
            )
    else:
        print("Skipping plotting suite example as df_diagnostic is empty.")
