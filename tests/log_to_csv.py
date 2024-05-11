from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import tensorflow as tf
import csv
from pathlib import Path
import pandas as pd
import os
import argparse

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='logs/', help='Directory containing the logs')
    parser.add_argument('--output_dir', type=str, default='output_logs/', help='Directory to save the csv file')
    parser.add_argument('--name', type=str, default='csv_output', help='Name of the csv file')
    return parser.parse_args()

def log_to_csv(log_file, destination, name):
    # Create the .csv file
    csv_path = destination + name + ".csv"
    Path(csv_path).touch()

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        accumulator = EventAccumulator(log_file)
        accumulator.Reload()
        tags = accumulator.Tags()['scalars']
        tags = ['step'] + tags
        
        data = [[] for i in range(len(tags))]
        for i in range(len(tags)):
            data[i].append(tags[i])

        for e in tf.compat.v1.train.summary_iterator(log_file):
            datapoint = e.summary.value
            for i in range(len(datapoint)):
                tag=datapoint[i].tag
                value=datapoint[i].simple_value
                
                #find the index of the tag in the tags list
                index = tags.index(tag)
                #add the value to the correct collum
                data[index].append(value)
        
        old = 0
        for e in tf.compat.v1.train.summary_iterator(log_file):
            step = e.step
            if step != old:
                index = tags.index('step')
                data[index].append(step)
                old = step
        
        last_collum = []
        for i in range(0, len(data)):
            #append the last ellement in each row to the last_collum list
            #some will be patted with their own last element even though they are empty
            last_collum.append(data[i][-1])
        
        data = zip(*data)
        writer.writerows(data)
        writer.writerow(last_collum)
        #close the file
        file.close()
        
    print(f"Data extracted to {csv_path}")

if __name__=='__main__':
    args = args()
    out_dir = args.output_dir
    log_dir = args.log_dir
    #out_dir = log_dir
    file_name = args.name

    swimmer_test_path = "tests/Swimmer/Combined/Swimmer_eps_LSTM_alpha_AE_fit_1/logs/plot_test/"
    out_dir = swimmer_test_path.replace("plot_test", "plot_test_out")
    log_dir = swimmer_test_path
    file_name = ""
    
    #make sure dst_dir exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    #loop through all folders in log_dir
    for subdir, dirs, files in os.walk(log_dir):
        dir_name = os.path.basename(subdir)
        # print(f"dir name: {dir_name}, out_dir: {out_dir}")
        # if dir_name == out_dir:
        #     continue    #skip the output directory
        for file in files: #only one should exist
            # print(os.path.join(subdir, file))
            log_to_csv(os.path.join(subdir, file), out_dir, file_name + dir_name)
