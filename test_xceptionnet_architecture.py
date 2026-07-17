#!/usr/bin/env python3
"""
Quick test: Run a dummy tensor through XceptionNet to verify shapes.
"""

import torch
from face_id.models._deepfake_architecture import XceptionNet

# Create model
model = XceptionNet(num_classes=2)
model.eval()  # Evaluation mode (disables dropout, etc.)

print("Testing XceptionNet architecture...\n")

# Create a dummy batch of images
batch_size = 4
dummy_input = torch.randn(batch_size, 3, 224, 224)
print(f"Input shape: {dummy_input.shape}")

# Run through network
with torch.no_grad():
    output = model(dummy_input)

print(f"Output shape: {output.shape}")
print(f"Expected: torch.Size([{batch_size}, 2])")

if output.shape == torch.Size([batch_size, 2]):
    print("\n✓ Architecture test PASSED!")
    print(f"Sample output (logits): {output[0]}")
else:
    print("\n✗ Architecture test FAILED!")
    print(f"Expected shape: torch.Size([{batch_size}, 2]), got {output.shape}")
