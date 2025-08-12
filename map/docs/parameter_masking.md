# fNP Parameter Masking System

## Overview

The `fNP.py` module implements a sophisticated parameter masking system that allows users to control which parameters are trainable during optimization and which remain fixed. This is controlled through the `free_mask` configuration in `fNPconfig.yaml`.

## How It Works

### Configuration File (`fNPconfig.yaml`)

Each flavor in the configuration has two key arrays:

- `init_params`: Initial values for the parameters
- `free_mask`: Boolean array indicating which parameters are trainable (`true`) or fixed (`false`)

```yaml
flavors:
  u:
    init_params: [0.25, 0.15, 0.12, 0.10, 0.20, 0.18, 0.08, 0.14, 0.13, 0.11, 0.09]
    free_mask:   [true, true, false, true, true, false, true, true, true, false, true]
    #             N1   α1    σ1     λ    N1B   N1C    λ2   α2    α3    σ2    σ3
    #             ✓    ✓     ✗      ✓    ✓     ✗      ✓    ✓     ✓     ✗     ✓
```

### Implementation Details

1. **Parameter Storage**: Each flavor module stores parameters in two components:
   - `fixed_params`: Non-trainable parameters (stored as buffers)
   - `free_params`: Trainable parameters (stored as nn.Parameter)

2. **Gradient Masking**: A gradient hook ensures only free parameters receive gradients:

   ```python
   self.free_params.register_hook(lambda grad: grad * mask)
   ```

3. **Parameter Reconstruction**: The full parameter tensor is reconstructed as:

   ```python
   full_params = fixed_params + free_params
   ```

## Usage Examples

### Basic Parameter Analysis

```python
from modules.fNP import fNP
import yaml

# Load configuration
with open('inputs/fNPconfig.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create model
model = fNP(config)

# Print parameter summary
model.print_parameter_summary()

# Get detailed parameter info
info = model.get_parameter_info()
print(f"Truly trainable: {info['truly_trainable_parameters']}")
print(f"Total parameters: {info['total_parameters']}")
```

### Working with Only Trainable Parameters

```python
# Extract only trainable parameters (useful for optimization)
trainable_params = model.get_trainable_parameters_dict()

# Save trainable parameters
torch.save(trainable_params, 'fitted_params.pth')

# Load and apply trainable parameters
loaded_params = torch.load('fitted_params.pth')
model.set_trainable_parameters_dict(loaded_params)
```

### PyTorch Optimization

```python
# Create optimizer with all parameters (masking handled internally)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# During training, fixed parameters won't update due to gradient masking
for batch in dataloader:
    optimizer.zero_grad()
    loss = compute_loss(model, batch)
    loss.backward()  # Gradients zeroed for fixed parameters
    optimizer.step()  # Only free parameters update
```

## Parameter Counts

The system reports different parameter counts:

1. **Total Parameters**: All parameters in the model (fixed + trainable)
2. **PyTorch Trainable Parameters**: Parameters with `requires_grad=True` (may include masked ones)
3. **Truly Trainable Parameters**: Parameters that actually update during training (considering masks)

## Testing Parameter Masking

Use the included test script to verify masking works:

```bash
python test_parameter_masking.py
```

This script demonstrates:

- How different `free_mask` configurations affect trainability
- Gradient analysis showing which parameters receive gradients
- Parameter change verification after optimization steps

## Benefits for TMD Fitting

1. **Physics Constraints**: Fix parameters based on physical requirements
2. **Staged Fitting**: Fix some parameters while fitting others
3. **Parameter Transfer**: Use fitted parameters from one model in another
4. **Reduced Overfitting**: Limit degrees of freedom during optimization
5. **Computational Efficiency**: Fewer parameters to optimize

## Configuration Examples

### All Parameters Trainable

```yaml
flavors:
  u:
    init_params: [0.25, 0.15, 0.12]
    free_mask:   [true, true, true]
```

### Some Parameters Fixed

```yaml
flavors:
  u:
    init_params: [0.25, 0.15, 0.12]
    free_mask:   [true, false, true]  # Fix second parameter
```

## Technical Notes

- Fixed parameters are stored as buffers and don't appear in `model.parameters()`
- Gradient hooks ensure computational efficiency by zeroing unwanted gradients
- The masking system is fully compatible with PyTorch's autograd system
- Parameter tensors maintain proper device placement (CPU/GPU)
- The system supports dynamic reconfiguration of masks if needed
