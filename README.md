# pytorch_training_optimization_using_tensordict_memory_mapping

Optimizing PyTorch training by wrapping torch.utils.data.Dataset with tensordict.TensorDict.MemoryMappedTensor mapped, pinned,
and loaded onto an Nvidia GPU and inputting TensorDict(Dataset) into torch.utils.data.DataLoader--to boost model training speed.

## Boost PyTorch Model Training Speed:
![subplots_demo](https://github.com/user-attachments/assets/203bbc6e-446e-442c-ac38-5b022352a35f)


To run the demo:
````
git clone https://github.com/OriYarden/pytorch_training_optimization_using_tensordict_memory_mapping
cd pytorch_training_optimization_using_tensordict_memory_mapping
python run_demo.py
````

Training 1 Epoch via torch.utils.data.Dataset:

![demo_dataloader](https://github.com/user-attachments/assets/612806d8-3a8a-442c-8c2a-3ff2232d935b)


Training 1 Epoch via tensordict.TensorDict.MemoryMappedTensor(torch.utils.data.Dataset):

![demo_td_dataloader](https://github.com/user-attachments/assets/f580bd2f-3352-4ead-a7e4-35387e0d4f71)

TensorDict Memory Mapping boosts training speed.

The initial wrapping runtime is approximately equal to 1 epoch of torch.utils.data.Dataset:

![demo_td_wrapper](https://github.com/user-attachments/assets/d56f0384-b9d0-4356-91aa-dc86808c0f33)











