# Multiple GPUs Architecture - Enumerating (in Parallel) Stacked TensorDict Memory Mapped Tensors as a Super Batch:

![Picture1](https://github.com/user-attachments/assets/fead7e40-6ade-4a6d-8b86-4f959a1eda9f)

The batch dimension size of a ````super_batch```` is ````batch_size```` x ````len(torch.cuda.device_count()) - 1````.



