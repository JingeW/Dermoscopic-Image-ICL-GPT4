"""
Stats for AI generated content
"""

import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import argparse
import numpy as np

def create_label(bn_folder, mm_folder):
    image_data = []
    for image in os.listdir(bn_folder):
        image_data.append({'Image': image, 'Label': 'Benign'})
        
    for image in os.listdir(mm_folder):
        image_data.append({'Image': image, 'Label': 'Melanoma'})
    
    df = pd.DataFrame(image_data)
    df = df.sort_values(by='Image')
    return df

def load_and_merge_results(res_paths):
    dfs = [pd.read_csv(path)[['Image', 'Classification']] for path in res_paths]
    merged_df = dfs[0]
    for i, df in enumerate(dfs[1:], start=2):
        merged_df = merged_df.merge(df, on='Image', suffixes=(f'_run{i-1}', f'_run{i}'))
    merged_df.rename(columns={'Classification': f'Classification_run{len(dfs)}'}, inplace=True)
    return merged_df

def calculate_consensus(merged_df, reps):
    consensus_col = merged_df[[f'Classification_run{i}' for i in range(1, reps + 1)]].mode(axis=1)[0]
    merged_df['Consensus'] = consensus_col
    return merged_df

def calculate_metrics(true_labels, predicted_labels):
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    return accuracy, specificity, sensitivity

def get_wrong_in_all_runs(merged_df, reps):
    wrong_in_all_runs = []
    for _, row in merged_df.iterrows():
        if all(row[f'Classification_run{i}'] == row['Classification_run1'] for i in range(2, reps + 1)):
            if row['Classification_run1'] != row['Label']:
                wrong_in_all_runs.append(row['Image'])
    return wrong_in_all_runs

def parse_args():
    parser = argparse.ArgumentParser(description="Result analysis")
    parser.add_argument('--k', type=int, default=0, help='Number of examples for few-shot learning')
    parser.add_argument('--knn', type=bool, default=False, help='Use knn to pick example or not. Default is False.')
    parser.add_argument('--reps', type=int, default=5, help='Total repetitions')
    parser.add_argument('--process', nargs='+', default=None, help='Process methods[query, examples], e.g., inpaint, crop. Pass multiple values separated by spaces.')
    parser.add_argument('--prompt_version', type=str, default='v3.0', help='Version of the text prompts used')
    parser.add_argument('--res_dir', type=str, default='./result', help='Folder of resuls')
    return parser.parse_args()

def main():
    args = parse_args()
    # Print settings to ensure everything is loaded correctly
    print(f"Number of Examples (k): {args.k}")
    print(f"Use KNN: {args.knn}" )
    print(f'Total reptitions: {args.reps}')
    print(f'Image preprocessing: {args.process}')
    print(f'Prompt version: {args.prompt_version}')
    print(f'Result directory: {args.prompt_version}')
    print(f'\n')
        
    k = args.k
    knn = args.knn
    reps = args.reps
    process = args.process
    prompt_version = args.prompt_version
    res_dir = args.res_dir
    
    #=========================================
    # Task configuration
    if k == 0:
        task = f'{k}_shot_{prompt_version}{("_" + process[0] + "_" + process[1]) if process else ""}'
    else:
        task =  f'{k}_shot_{prompt_version}_{"KNN" if knn else "Random"}{("_" + process[0] + "_" + process[1]) if process else ""}'
    bn_dir = f'./data/bn_resized_label{("_" + process[0]) if process else ""}'
    mm_dir = f'./data/mm_resized_label{("_" + process[0]) if process else ""}'
    # bn_dir = './data/bn_test'
    # mm_dir = './data/mm_test'
    
    print(f"Task: {task}")
    res_paths = [os.path.join(res_dir, f'{task}', f'rep{i}', f'{task}.csv') for i in range(1, reps + 1)]
    print('------------------------------------')

    merged_df = load_and_merge_results(res_paths)
    merged_df = calculate_consensus(merged_df, reps)

    gt_df = create_label(bn_dir, mm_dir)
    merged_df = merged_df.merge(gt_df, on='Image')  # Merge ground truth labels

    # Accumulate metrics for each run
    accuracies, specificities, sensitivities = [], [], []
    binary_true_labels = [0 if label == 'Benign' else 1 for label in merged_df['Label'] if label != 'Unknown']
    
    for i in range(reps):
        binary_predicted_labels = [0 if label == 'Benign' else 1 for label, gt_label in zip(merged_df[f'Classification_run{i+1}'], merged_df['Label']) if gt_label != 'Unknown']
        accuracy, specificity, sensitivity = calculate_metrics(binary_true_labels, binary_predicted_labels)
        print(f'Run {i+1}:')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Specificity: {specificity:.4f}')
        print(f'Sensitivity: {sensitivity:.4f}')
        print('------------------------------------')
        
        accuracies.append(accuracy)
        specificities.append(specificity)
        sensitivities.append(sensitivity)

    # Calculate and print average metrics and standard deviation
    avg_accuracy, std_accuracy = np.mean(accuracies), np.std(accuracies)
    avg_specificity, std_specificity = np.mean(specificities), np.std(specificities)
    avg_sensitivity, std_sensitivity = np.mean(sensitivities), np.std(sensitivities)
    
    print('Average Metrics:')
    print(f'Accuracy: {avg_accuracy:.4f} +/- {std_accuracy:.4f}')
    print(f'Specificity: {avg_specificity:.4f} +/- {std_specificity:.4f}')
    print(f'Sensitivity: {avg_sensitivity:.4f} +/- {std_sensitivity:.4f}')
    print('------------------------------------')

    binary_predicted_labels_consensus = [0 if label == 'Benign' else 1 for label, gt_label in zip(merged_df['Consensus'], merged_df['Label']) if gt_label != 'Unknown']
    accuracy, specificity, sensitivity = calculate_metrics(binary_true_labels, binary_predicted_labels_consensus)
    print('Consensus:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Specificity: {specificity:.4f}')
    print(f'Sensitivity: {sensitivity:.4f}')
    print('------------------------------------')

    wrong_in_all_runs = get_wrong_in_all_runs(merged_df[merged_df['Label'] != 'Unknown'], reps)
    print('Images wrong in all runs:', wrong_in_all_runs)

if __name__ == "__main__":
    main()