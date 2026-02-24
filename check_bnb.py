import torch
try:
    import bitsandbytes as bnb
    print(f"bitsandbytes version: {bnb.__version__}")
    
    # Check if CUDA is available for bitsandbytes
    # This usually triggers the load of CUDA kernels
    from bitsandbytes.nn import Linear4bit
    print("Successfully imported Linear4bit from bitsandbytes.")
    
    # Simple test to see if we can create a 4-bit layer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        model = torch.nn.Linear(10, 10).to(device)
        print("Standard Linear layer moved to CUDA.")
        
        # Test 4-bit quantization (this is what we need for QLoRA)
        try:
            from peft import prepare_model_for_kbit_training
            print("Successfully imported PEFT kbit preparation.")
        except ImportError:
            print("PEFT not installed yet or not found.")
            
except Exception as e:
    print(f"Error checking bitsandbytes: {e}")
    import traceback
    traceback.print_exc()
