import os
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import yaml

def main():
    # Paths
    dataset_yaml = os.path.abspath('../datasets/baches/data.yaml')
    model_dir = os.path.abspath('../models')
    model_path = os.path.join(model_dir, '../models/yolo11s.pt')
    results_dir = os.path.abspath('./results')
    os.makedirs(results_dir, exist_ok=True)

    # Load dataset info
    with open(dataset_yaml, 'r') as f:
        data_cfg = yaml.safe_load(f)
    print('Classes:', data_cfg.get('names', []))

    # Load model
    model = YOLO(model_path)

    # Train
    results = model.train(
        data=dataset_yaml,
        epochs=50,
        imgsz=640,
        batch=8,
        project=results_dir,
        name='yolo11m_baches',
        exist_ok=True
    )

    # Plot metrics
    metrics = results.metrics
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['epoch'], metrics['train/box_loss'], label='Train Box Loss')
    plt.plot(metrics['epoch'], metrics['val/box_loss'], label='Val Box Loss')
    plt.plot(metrics['epoch'], metrics['metrics/mAP_0.5'], label='mAP@0.5')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('YOLOv8 Training Metrics')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(results_dir, 'training_metrics.png'))
    plt.show()

    # Test
    test_results = model.val(data=dataset_yaml, split='test')
    print('Test Results:', test_results)

    # Save test metrics
    with open(os.path.join(results_dir, 'test_metrics.yaml'), 'w') as f:
        yaml.dump(test_results, f)

if __name__ == '__main__':
    main()
