# pytorch_training_optimization_using_tensordict_memory_mapping

Optimizing PyTorch training by wrapping ````torch.utils.data.Dataset```` with ````tensordict.TensorDict.MemoryMappedTensor```` mapped, pinned,
and loaded onto an Nvidia GPU and inputting ````TensorDict(Dataset)```` into ````torch.utils.data.DataLoader````--to boost model training speed.

# Boost PyTorch Model Training Speed:
![subplots_demo](https://github.com/user-attachments/assets/203bbc6e-446e-442c-ac38-5b022352a35f)


To run the demo:
````
git clone https://github.com/OriYarden/pytorch_training_optimization_using_tensordict_memory_mapping
cd pytorch_training_optimization_using_tensordict_memory_mapping
python run_demo.py
````

# Visualizing ````tensordict_packages```` Enwrapment:

![image](https://github.com/user-attachments/assets/4844201c-2a38-4468-abb0-4c3492e097a8)


# Visualizing PyTorch TensorDict Memory Mapped Tensors Speed Advantage:
(and what ````run_demo.py```` looks like in gifs)

### PyTorch Model Training BASELINE - Control Condition
````
torch.utils.data.Dataset # Training 1 Epoch:
````

![demo_dataloader](https://github.com/user-attachments/assets/612806d8-3a8a-442c-8c2a-3ff2232d935b)

### PyTorch Model Training TEST - Experimental Condition
````
tensordict.TensorDict.MemoryMappedTensor(torch.utils.data.Dataset) # Training 1 Epoch:
````

![demo_td_dataloader](https://github.com/user-attachments/assets/f580bd2f-3352-4ead-a7e4-35387e0d4f71)


````torch.utils.data.Dataset````'s POV:

![lamborghini-race-car](https://github.com/user-attachments/assets/d5e4d7f9-e69f-478c-ab29-9018e629b904)


The only thing you have to change in your code (along with potentially a few other minor changes, see comments in code):

````
ds = Dataset() # <--- potentially requires minor changes in __getitem__ method
ds = dataset_to_tensordict( # <--- Wraps here, this must be added into your existing code (from tensordict_packages).
    ds=ds,
    DEVICE=DEVICE,
)
loader = DataLoader() # <--- requires inputting Collate_Fn wrapper (from tensordict_packages).
# That's it! Just two lines of code.
````

# Concluding Remarks:

The TensorDict Memory Mapping tools that I've provided in ````tensordict_packages```` boosts PyTorch model training speed.

However, the initial ````tensordict_packages```` wrapping runtime is approximately equal to 1 epoch of ````torch.utils.data.Dataset````:

![demo_td_wrapper](https://github.com/user-attachments/assets/d56f0384-b9d0-4356-91aa-dc86808c0f33)

So there may not be a scenario in which ````tensordict_packages```` can benefit PyTorch model inferencing alone.

Still, PyTorch model training speed can be improved by orders of magnitude when using ````tensordict_packages````, and therefore,
we should make the most out of the Nvidia GPU resources (i.e. memory) available so that we can speed up PyTorch model training time,
reduce PyTorch model training cost, and shorten the gap between initially developing PyTorch models and having PyTorch models in production.

And with the current AI boom, where LLMs and text-to-video PyTorch models require months of training, we can save time, resources, and
Nvidia GPUs via ````tensordict_packages````'s ability to leverage ````TensorDict```` and ````MemoryMappedTensor````s with ````torch.utils.data.DataLoader````.





