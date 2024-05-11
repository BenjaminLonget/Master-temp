import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_csv_files(folder_path, x_column, y_column, plot_name):
    # Get list of CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    plt.rcParams.update({'font.size': 14})
    # Iterate over each CSV file
    for csv_file in csv_files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(os.path.join(folder_path, csv_file))
        
        # Plot the data from the CSV file
        plt.plot(df[x_column], df[y_column], label=csv_file.replace('.csv', ''), linewidth=2.0)
    
    # Add labels and legend
    plt.grid()
    plt.title(f'{plot_name} {y_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend()
    
    # Show the plot
    plt.show()

plot_name = "Swimmer comparison"
folder_path = 'tests/Swimmer/Combined/Swimmer_eps_LSTM_alpha_AE_fit_1/logs/plot_test_out2'
x_column = 'step'
y_column = 'train/mean_evaluation_reward'
plot_csv_files(folder_path, x_column, y_column, plot_name)