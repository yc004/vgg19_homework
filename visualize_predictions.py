import os
import random
import matplotlib.pyplot as plt
from PIL import Image
from mmpretrain.apis import ImageClassificationInferencer

def visualize_predictions(config_path, checkpoint_path, data_root, num_samples=9):
    # Initialize the inferencer
    try:
        inferencer = ImageClassificationInferencer(model=config_path, pretrained=checkpoint_path, device='cuda')
    except Exception as e:
        print(f"Failed to init inferencer on cuda, trying cpu. Error: {e}")
        inferencer = ImageClassificationInferencer(model=config_path, pretrained=checkpoint_path, device='cpu')

    # Collect all validation images
    val_root = os.path.join(data_root, 'val')
    classes = sorted(os.listdir(val_root))
    all_images = []
    
    for cls_name in classes:
        cls_dir = os.path.join(val_root, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for img_name in os.listdir(cls_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_images.append({
                    'path': os.path.join(cls_dir, img_name),
                    'gt_label': cls_name
                })

    if not all_images:
        print("No images found in validation set.")
        return

    # Sample images
    samples = random.sample(all_images, min(num_samples, len(all_images)))

    # Setup plot
    rows = int(num_samples ** 0.5)
    cols = (num_samples + rows - 1) // rows
    plt.figure(figsize=(15, 5 * rows))

    print(f"Running inference on {len(samples)} images...")
    
    for i, sample in enumerate(samples):
        img_path = sample['path']
        gt_label = sample['gt_label']
        
        # Inference
        result = inferencer(img_path)[0]
        pred_label = result['pred_class']
        pred_score = result['pred_score']
        
        # Read image for display
        img = Image.open(img_path).convert('RGB')

        # Plot
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        
        color = 'green' if pred_label == gt_label else 'red'
        title = f"GT: {gt_label}\nPred: {pred_label} ({pred_score:.2f})"
        
        plt.title(title, color=color, fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    output_file = 'prediction_results.png'
    plt.savefig(output_file)
    print(f"Prediction visualization saved to {output_file}")

if __name__ == '__main__':
    config_file = 'configs/vgg19_intel.py'
    # Use epoch 4 as epoch 5 save failed
    checkpoint_file = 'work_dirs/vgg19_intel/epoch_4.pth'
    data_root = 'data/intel_image'
    
    if not os.path.exists(checkpoint_file):
        print(f"Checkpoint {checkpoint_file} not found. Using epoch_5.pth if available.")
        checkpoint_file = 'work_dirs/vgg19_intel/epoch_5.pth'

    if os.path.exists(config_file) and os.path.exists(checkpoint_file):
        visualize_predictions(config_file, checkpoint_file, data_root)
    else:
        print("Config or checkpoint not found.")
