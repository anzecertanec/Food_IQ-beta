import tkinter as tk
from tkinter import filedialog, ttk, IntVar, messagebox
from tkinter.simpledialog import askstring
import pandas as pd
import statsmodels.api as sm
import os
from joblib import dump
import itertools
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random

class CSVHeaderSelector:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CSV Header Selector")
        self.root.state('zoomed')
        self.filepath = None
        self.df = None
        self.dependent_var = tk.StringVar()
        self.is_centered = False

        self.delimiter_label = tk.Label(self.root, text="Select Delimiter:")
        self.delimiter_label.pack(pady=10)

        self.delimiter_combobox = ttk.Combobox(self.root, values=[";", ",", ".", "|", "\t", " "], state="readonly")
        self.delimiter_combobox.set(";")
        self.delimiter_combobox.pack(pady=10)

        self.upload_btn = tk.Button(self.root, text="Upload CSV", command=self.upload_csv)
        self.upload_btn.pack(pady=20)

        self.root.mainloop()
    def add_dummies_callback(self):
        # Check if dummy listbox already exists, if yes, remove it
        if hasattr(self, 'dummies_listbox'):
            self.dummies_listbox.destroy()
            self.dummies_listbox_label.destroy()
            self.confirm_dummies_btn.destroy()
            self.cancel_dummies_btn.destroy()

        # Create a new listbox for dummy variable selection
        self.dummies_listbox_label = tk.Label(self.headers_win, text="Select Columns for Dummies:")
        self.dummies_listbox_label.place(x=550, y=53)

        self.dummies_listbox = tk.Listbox(self.headers_win, selectmode=tk.MULTIPLE)
        for item in self.df_core.columns:
            self.dummies_listbox.insert(tk.END, item)
        self.dummies_listbox.place(x=550, y=100, height=350, width=200)

        # Add a confirm button to finalize dummy variable creation
        self.confirm_dummies_btn = tk.Button(self.headers_win, text="Confirm Dummies", command=self.create_dummies)
        self.confirm_dummies_btn.place(x=500, y=460)

        self.cancel_dummies_btn = tk.Button(self.headers_win, text="prekliči", command=self.cancel_dummies)
        self.cancel_dummies_btn.place(x=650, y=460)
    def cancel_dummies(self):
        self.dummies_listbox.destroy()
        self.dummies_listbox_label.destroy()
        self.confirm_dummies_btn.destroy()
        self.cancel_dummies_btn.destroy()
    def create_dummies(self):

        # Load bulder_memory1.csv if it exists
        self.df = self.df_core

        # Get selected columns for dummy variable creation
        selected_columns = [self.dummies_listbox.get(i) for i in self.dummies_listbox.curselection()]

        # Store the selected columns as an instance variable for later use
        self.selected_columns_for_dummies = selected_columns

        total_new_columns = 0  # Initialize counter for new columns

        # For each selected column, count potential dummy variables (without actually creating them yet)
        for column in selected_columns:
            unique_values = len(self.df[column].unique())
            total_new_columns += unique_values - 1  # We subtract 1 because drop_first=True

        # If the total number of new columns is greater than 15, prompt the user with a warning
        if total_new_columns > 15:
            response = messagebox.askyesno("Warning",
                                           "This will add over 15 columns to your CSV which will most likely "
                                           "hurt the chances of providing statistically significant models. "
                                           "Would you still like to continue?")
            if not response:  # If user selects "No", return and do not add the columns
                return

        # If user selects "Yes" or total_new_columns <= 15, proceed with creating dummy variables
        for column in selected_columns:
            dummies = pd.get_dummies(self.df[column], prefix=column, drop_first=True).astype(
                int)  # Convert to integer type
            self.df = pd.concat([self.df, dummies], axis=1)

        # Save the modified dataframe to 'bulder_memory1.csv'
        self.df.to_csv('bulder_memory1.csv', index=False)

        # Provide feedback to the user
        messagebox.showinfo("Success", "Dummy variables created and saved to 'bulder_memory1.csv'")
        self.dummies_listbox.destroy()
        self.dummies_listbox_label.destroy()
        self.confirm_dummies_btn.destroy()
        self.cancel_dummies_btn.destroy()
        self.find_non_numeric()

    def evaluate_combinations(self):
        messagebox.showinfo("OPOZORILO",
                            "ta funciaj vkljucuje komplicirano matematiko zato lahko traja nekaj casa prosim da ne klikas drugih gumbov im pocakas 1 do 2 minuti, da vse izračunam še posebaj če list neignoriranih vkljucuje vec kot 10 spremelivk. zdar prosm klikni vredu")

        def powerset(lst):
            # Returns a generator for all the subsets of lst
            for i in range(len(lst) + 1):
                for subset in itertools.combinations(lst, i):
                    yield subset

        best_r2 = 0
        best_combination = []
        dependent = self.dropdown1.get()

        ignored = [self.ignore_listbox.get(i) for i in self.ignore_listbox.curselection()]

        # Extract the dummy columns
        dummy_columns = []
        if hasattr(self, 'selected_columns_for_dummies'):
            dummy_columns = [col for col in self.df.columns if col.startswith(
                tuple(self.selected_columns_for_dummies)) and col not in self.selected_columns_for_dummies]

        all_ignored = set(ignored)
        if hasattr(self, 'selected_columns_for_dummies'):
            all_ignored.update(self.selected_columns_for_dummies)

        predictors = [col for col in self.df.columns if col != dependent and col not in all_ignored]

        non_dummy_predictors = [col for col in predictors if col not in dummy_columns]

        # For each combination of non-dummy predictors
        for comb in powerset(non_dummy_predictors):
            # Skip combinations that include original dummy columns
            if hasattr(self, 'selected_columns_for_dummies') and any(
                    dummy_col in comb for dummy_col in self.selected_columns_for_dummies):
                continue

            # Add all dummy columns to current combination
            current_combination = list(comb) + dummy_columns

            X = self.df[current_combination]
            y = self.df[dependent]

            if X.empty:
                continue

            if self.is_centered:
                X = sm.add_constant(X)

            model = sm.OLS(y, X).fit()
            p_values = model.pvalues.drop("const", errors='ignore')

            # Check p-values for non-dummy columns
            non_dummy_p_values = [p_values[col] for col in comb if col in p_values]
            if not all(p < 0.05 for p in non_dummy_p_values):
                continue

            # Check p-values for dummy columns
            dummy_p_values = [p_values[col] for col in dummy_columns if col in p_values]
            if not all(p < 0.4 for p in dummy_p_values):
                continue

            # If both conditions above pass and R^2 is higher, update best combination
            if model.rsquared > best_r2:
                best_r2 = model.rsquared
                best_combination = current_combination

        messagebox.showinfo("Najbolsi model",
                            f"Najbolsi model vkljucuje te spremenivke {best_combination} in dosega R-squared: {best_r2} prosim klikni regresija")

        # Deselect all items in the listbox
        self.ignore_listbox.selection_clear(0, tk.END)

        # Iterate over all items in the listbox and select the ones not in best_combination
        for index, item in enumerate(self.ignore_listbox.get(0, tk.END)):
            if item not in best_combination:
                self.ignore_listbox.selection_set(index)

    def plot_model_fit(self):
        dependent = self.dropdown1.get()
        ignored = [self.ignore_listbox.get(i) for i in self.ignore_listbox.curselection()]
        X = self.df.drop(columns=[dependent] + ignored)
        y = self.df[dependent]

        if self.is_centered:
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
        else:
            model = sm.OLS(y, X).fit()

        predicted_values = model.predict(X)

        # Create a range for the x-axis
        x_range = range(1, len(y) + 1)

        # Create a new Figure
        fig = plt.Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(x_range, y, 'g-', label='Actual Values')
        ax.plot(x_range, predicted_values, 'b-', label='Predicted Values')
        ax.set_xlabel('Row Number')
        ax.set_ylabel(dependent)
        ax.set_title('Comparison of Actual vs Predicted Values')
        ax.legend()
        ax.grid(True)

        # Create a new window to display the graph
        graph_window = tk.Toplevel(self.headers_win)  # Toplevel creates a new window
        graph_window.title("Model Fit Visualization")

        # Embedding the figure in the new window
        canvas = FigureCanvasTkAgg(fig, master=graph_window)  # Setting master to the new window
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)  # Adjusting canvas widget to fit the new window
        canvas.draw()
    def remove_dummies(self):
        self.df = self.df_core
        messagebox.showinfo("remove","removed dummies")

    def toggle_centered(self):
        self.is_centered = not self.is_centered
        print("Centered:", self.is_centered)

    def upload_csv(self):
        self.filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.filepath:
            delimiter = self.delimiter_combobox.get()
            self.df = pd.read_csv(self.filepath, delimiter=delimiter)
            self.df_core = pd.read_csv(self.filepath, delimiter=delimiter)
            self.root.destroy()
            self.display_headers()

    def display_headers(self):
        self.headers_win = tk.Tk()
        self.headers_win.title("izberi spremenlivke")
        self.headers_win.state('zoomed')

        label1 = tk.Label(self.headers_win, text="odvisna spremenivka")
        label1.place(x=50, y=50)

        best_model_btn = tk.Button(self.headers_win, text="Find Best Model", command=self.evaluate_combinations)
        best_model_btn.place(x=100, y=250)

        self.dropdown1 = ttk.Combobox(self.headers_win, textvariable=self.dependent_var,
                                      values=self.df.columns.tolist())
        self.dropdown1.place(x=300, y=50, width=200)

        label2 = tk.Label(self.headers_win,
                          text="izberi spremenilvke ki jih želiš ignorirati\n (prosim izberi vse spremenlivke ki niso\n stevilke npr lokacija in datum)")
        label2.place(x=50, y=100)

        self.ignore_listbox = tk.Listbox(self.headers_win, selectmode=tk.MULTIPLE)
        for item in self.df.columns:
            self.ignore_listbox.insert(tk.END, item)
        self.ignore_listbox.place(x=300, y=100, height=350, width=200)

        self.centered_checkbox = tk.Checkbutton(self.headers_win, text="Centered", command=self.toggle_centered)
        self.centered_checkbox.place(x=180, y=200)

        run_regression_btn = tk.Button(self.headers_win, text="Regresija", command=self.run_regression)
        run_regression_btn.place(x=100, y=200)
        self.modify_csv_btn = tk.Button(self.headers_win, text="spremeni vejice v pike", command=self.modify_and_save_csv)
        self.modify_csv_btn.place(x=600, y=150)

        self.r_squared_text = tk.Text(self.headers_win, height=2, width=20)
        self.r_squared_text.place(x=1150, y=50)

        self.regression_output = tk.Text(self.headers_win, height=31, width=40)
        self.regression_output.place(x=800, y=50)

        self.export_btn = tk.Button(self.headers_win, text="Export Model", command=self.export_model, state=tk.DISABLED)
        self.export_btn.place(x=100, y=450)

        self.model_fit_btn = tk.Button(self.headers_win, text="Prikaži graf", command=self.plot_model_fit,
                                       state=tk.DISABLED)
        self.model_fit_btn.place(x=100, y=350)

        self.random_prediction_btn = tk.Button(self.headers_win, text="Naključna napoved", state=tk.DISABLED,
                                               command=self.random_prediction)
        self.random_prediction_btn.place(x=100, y=300)

        self.remove_outliers_btn = tk.Button(self.headers_win, text="Ignore Outliers", state=tk.DISABLED,
                                             command=self.remove_outliers_button_clicked)
        self.remove_outliers_btn.place(x=620, y=100)

        self.remove_dummies_btn = tk.Button(self.headers_win, text="remove duumies",
                                             command=self.remove_dummies)
        self.remove_dummies_btn.place(x=620, y=200)

        self.model_summary_btn = tk.Button(self.headers_win, text="poglej opis modela",state=tk.DISABLED, command=self.display_model_summary)
        self.model_summary_btn.place(x=100, y=400)

        self.back_btn = tk.Button(self.headers_win, text="Back", command=self.go_back_to_main)
        self.back_btn.place(x=650, y=650)

        self.back_btn = tk.Button(self.headers_win, text="Add dummies", command=self.add_dummies_callback)
        self.back_btn.place(x=620, y=50)
        self.find_non_numeric()
        self.headers_win.mainloop()

    def display_model_summary(self):

        summary_window = tk.Toplevel(self.headers_win)
        summary_window.title("Model Summary")

            # Add the model summary to the window using a Text widget for better formatting
        text_widget = tk.Text(summary_window, wrap=tk.WORD)
        text_widget.insert(tk.END, str(self.summodel.summary()))
        text_widget.pack(padx=12, pady=12, fill=tk.BOTH, expand=True)


    def find_non_numeric(self):
        # Iterate over each column in the dataframe
        for col in self.df.columns:
            # If any value in the column is non-numeric, select the corresponding header in the listbox
            if self.df[col].apply(lambda x: not isinstance(x, (int, float))).any():
                # Find the index of the column header in the listbox
                idx = self.ignore_listbox.get(0, tk.END).index(col)
                self.ignore_listbox.selection_set(idx)

    def go_back_to_main(self):
        self.headers_win.destroy()
        CSVHeaderSelector()

    def insert_colored_text(self, widget, text, color):
        tag = "color_" + color
        widget.tag_configure(tag, foreground=color)
        widget.insert(tk.END, text, tag)

    def modify_and_save_csv(self):
        # Copying the current DataFrame with ; as the delimiter
        self.df.to_csv("bulder_memory1.csv", sep=";", index=False)

        # Reading the file, replacing all occurrences of , with .
        with open("bulder_memory1.csv", "r") as file:
            content = file.read().replace(',', '.')

        # Writing the modified content back to the file
        with open("bulder_memory1.csv", "w") as file:
            file.write(content)

        # Updating self.df to read from the modified CSV
        self.df = pd.read_csv("bulder_memory1.csv", sep=";")
        messagebox.showwarning("opravljeno", "sedj so decimalke zapisane s pikami namesto vejicami (brez skrbi originalni ni spremenjen)")

    def run_regression(self):
        self.find_non_numeric()
        dependent = self.dropdown1.get()
        if not dependent:
            messagebox.showwarning("Warning", "Please select a dependent variable.")
            return


        ignored = [self.ignore_listbox.get(i) for i in self.ignore_listbox.curselection()]
        if hasattr(self, 'selected_columns_for_dummies'):
            # If the first element is a list, flatten the list
            if isinstance(self.selected_columns_for_dummies[0], list):
                flattened_dummies = [item for sublist in self.selected_columns_for_dummies for item in sublist]
            else:
                flattened_dummies = self.selected_columns_for_dummies

            # Add dummy columns to the ignored list
            ignored.extend(flattened_dummies)
        ignored = [col for col in ignored if col]  # Filter out any empty values

        X = self.df.drop(columns=[dependent] + ignored)
        y = self.df[dependent]

        if self.is_centered:
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
        else:
            model = sm.OLS(y, X).fit()

        p_values = model.pvalues.drop("const", errors='ignore')

        self.regression_output.delete(1.0, tk.END)
        self.regression_output.insert(tk.END,
                                      "P-vrednosti za spremenivke:\n prosim ignoriraj največjo rdečo\n izberi ga na seznamu levo nato klikni\nregresija\n")
        self.regression_output.insert(tk.END, "---------------------------------\n")
        for var, p_val in p_values.items():
            color = "black"
            if p_val > 0.10:
                color = "red"
            elif p_val > 0.05:
                color = "orange"
            self.insert_colored_text(self.regression_output, f"{var}: {p_val:.5f}\n", color)

        r_squared_val = model.rsquared
        self.r_squared_text.delete(1.0, tk.END)
        self.r_squared_text.insert(tk.END, f"R-squared: {r_squared_val:.5f}")
        self.summodel= model

        self.export_btn.config(state=tk.NORMAL)
        self.model_fit_btn.config(state=tk.NORMAL)
        self.random_prediction_btn.config(state=tk.NORMAL)
        self.remove_outliers_btn.config(state=tk.NORMAL)
        self.model_summary_btn.config(state=tk.NORMAL)


    def random_prediction(self):
        dependent = self.dropdown1.get()
        if not dependent:
            messagebox.showwarning("Warning", "Please select a dependent variable.")
            return

        ignored = [self.ignore_listbox.get(i) for i in self.ignore_listbox.curselection()]

        # Identify the original columns from which dummies were created
        dummy_origin_columns = [col.split("_")[0] for col in self.df.columns if "_" in col]
        # Add these columns to the ignored list
        ignored.extend(dummy_origin_columns)

        # Create the model first so we can get the order of predictors
        X = self.df.drop(columns=[dependent] + ignored)
        y = self.df[dependent]

        if self.is_centered:
            X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()

        # Now, get the order of predictors from the model
        predictors_order = [pred for pred in model.model.exog_names if pred != 'const']

        # Get random values
        random_values = {}

        # Identify dummy columns and group them by their original column name
        dummy_groups = {}
        for predictor in predictors_order:
            original_column = predictor.split("_")[0]
            if "_" in predictor:
                if original_column not in dummy_groups:
                    dummy_groups[original_column] = []
                dummy_groups[original_column].append(predictor)
            else:
                random_row = self.df.sample()
                random_values[predictor] = random_row[predictor].values[0]

        # For each group of dummy columns, randomly select one to be 1 and set others to 0
        for original_column, dummies in dummy_groups.items():
            chosen_dummy = random.choice(dummies)
            for dummy in dummies:
                random_values[dummy] = 1 if dummy == chosen_dummy else 0

        x_values_df = pd.DataFrame([random_values])

        # If model has a constant term, then add it
        if 'const' in model.params:
            x_values_df = sm.add_constant(x_values_df, has_constant='add')

        # Ensure columns are in the correct order for prediction
        x_values_df = x_values_df[model.model.exog_names]

        # Perform prediction
        prediction = model.predict(x_values_df)[0]

        messagebox.showinfo("Random Prediction",
                            f"Predicted value: {round(prediction)}\nUsing random values: {random_values}")

    def remove_outliers_button_clicked(self):
        dependent = self.dropdown1.get()
        ignored = [self.ignore_listbox.get(i) for i in self.ignore_listbox.curselection()]

        # Create the model first so we can get the order of predictors
        X = self.df.drop(columns=[dependent] + ignored)

        if self.is_centered:
            X = (X - X.mean())
        X = sm.add_constant(X)
        y = self.df[dependent]
        model = sm.OLS(y, X).fit()

        # Calculate deviations and the standard deviation
        predictions = model.predict(X)
        deviations = y - predictions
        std_dev = deviations.std()

        # Filter rows and write to the new CSV
        filtered_rows = self.df[(deviations < 3 * std_dev) & (deviations > -3 * std_dev)]
        ignored_rows_count = len(self.df) - len(filtered_rows)
        self.df = filtered_rows
        filtered_rows.to_csv("bulder_memory.csv", index=False)
        self.df = pd.read_csv("bulder_memory.csv")

        # Show the message box
        messagebox.showinfo("Info", f"{ignored_rows_count} rows ignored.")

    def export_model(self):
        dependent = self.dropdown1.get()
        ignored = [self.ignore_listbox.get(i) for i in self.ignore_listbox.curselection()]
        X = self.df.drop(columns=[dependent] + ignored)
        y = self.df[dependent]

        if self.is_centered:
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            cen_status = "cen"
        else:
            model = sm.OLS(y, X).fit()
            cen_status = "uncen"

        filename = os.path.basename(self.filepath).replace(".csv", "")
        predictors = "-".join(X.columns)
        r_squared_val = model.rsquared

        # Prompt user for a name
        user_name = askstring("Input", "Please enter a name for the model:")

        # If user cancels the dialog or enters an empty string, use a default name
        if not user_name:
            user_name = "default"

        # Create the output path
        output_path = f"{user_name}-({filename})-({predictors})-(r2={r_squared_val:.5f})-({cen_status}).joblib"

        dump(model, output_path)
        messagebox.showinfo("Exported", f"Model exported as {output_path}")


if __name__ == "__main__":
    app = CSVHeaderSelector()
