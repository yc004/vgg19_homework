import json
import matplotlib.pyplot as plt
import os

def visualize_training_results(log_file):
    train_epochs = []
    train_loss = []
    val_epochs = []
    val_acc = []

    print(f"Reading log file: {log_file}")
    with open(log_file, 'r') as f:
        for line in f:
            log = json.loads(line)
            
            # Training log
            if 'loss' in log:
                # Assuming 'epoch' is available in training logs
                if 'epoch' in log:
                    train_epochs.append(log['epoch'])
                    train_loss.append(log['loss'])
            
            # Validation log
            if 'accuracy/top1' in log:
                # Validation logs might use 'step' which corresponds to epoch in this config
                if 'step' in log:
                    val_epochs.append(log['step'])
                    val_acc.append(log['accuracy/top1'])

    # Plotting
    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(train_epochs, train_loss, label='Train Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.legend()

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(val_epochs, val_acc, label='Val Accuracy', marker='s', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    output_file = 'training_results.png'
    plt.savefig(output_file)
    print(f"Visualization saved to {output_file}")

if __name__ == '__main__':
    # Automatically find the latest log file
    work_dir = 'work_dirs/vgg19_intel'
    # Find the latest timestamp directory
    timestamps = [d for d in os.listdir(work_dir) if os.path.isdir(os.path.join(work_dir, d))]
    if not timestamps:
        print("No log directories found.")
    else:
        latest_dir = max(timestamps)
        log_path = os.path.join(work_dir, latest_dir, 'vis_data', 'scalars.json')
        
        if os.path.exists(log_path):
            visualize_training_results(log_path)
        else:
            print(f"Log file not found at {log_path}")
