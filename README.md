# pytorch_training_optimization_using_tensordict_memory_mapping
Optimizing PyTorch Training using TensorDict Memory Mapping to Nvidia GPU.

````
python==3.9.13
torch==2.3.1
tensordict==0.5.0
````

To run the demo [NOTE: a mock dataset will be created in the current working directory]:
````
git clone https://github.com/OriYarden/pytorch_training_optimization_using_tensordict_memory_mapping
cd pytorch_training_optimization_using_tensordict_memory_mapping
python run_demo.py
````

Training 1 Epoch via torch.utils.data.Dataset:

![demo_dataloader](https://github.com/user-attachments/assets/612806d8-3a8a-442c-8c2a-3ff2232d935b)


Training 1 Epoch via tensordict.TensorDict.MemoryMappedTensor(Dataset):

![demo_td_dataloader](https://github.com/user-attachments/assets/f580bd2f-3352-4ead-a7e4-35387e0d4f71)

TensorDict Memory Mapping boosts training speed.

The initial wrapping runtime is approximately equal to 1 epoch of torch.utils.data.Dataset:

![demo_td_wrapper](https://github.com/user-attachments/assets/d56f0384-b9d0-4356-91aa-dc86808c0f33)









