from mmengine.config import Config
from mmengine.runner import Runner

def main():
    # Load the config
    cfg = Config.fromfile('configs/vgg19_intel.py')
    
    # Set the work directory
    cfg.work_dir = 'work_dirs/vgg19_intel'
    
    # Build the runner
    runner = Runner.from_cfg(cfg)
    
    # Start training
    runner.train()

if __name__ == '__main__':
    main()
