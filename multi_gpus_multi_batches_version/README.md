# Multiple GPUs Architecture - Enumerating (in Parallel) Stacked TensorDict Memory Mapped Tensors as a Super Batch:

![Picture1](https://github.com/user-attachments/assets/fead7e40-6ade-4a6d-8b86-4f959a1eda9f)

The batch dimension size of a ````super_batch```` is ````batch_size```` multiplied by the number of memory mapped Nvidia GPUs (e.g. ````len(torch.cuda.device_count()) - 1```` in ````run_demo.py````).

NOTE: If using only one Nvidia GPU (e.g. ````MEMMAP_DEVICES = ['cuda:0']```` in ````run_demo.py````) then this version is equivalent to the base version of this repo which implements only a single Nvidia GPU for ````MemoryMappedTensor````s.


To run the demo:
````
git clone https://github.com/OriYarden/pytorch_training_optimization_using_tensordict_memory_mapping
cd pytorch_training_optimization_using_tensordict_memory_mapping/multi_gpus_multi_batches_version
python run_demo.py
````

However, there are no differences in the ````tensordict_packages```` for ````multi_gpus_multi_batches_version```` except for two new tools in
````tensordict_packages````'s ````utils_and_toolbox.py```` which allows in-parallel enumeration (````enumerate_loaders_in_parallel````) and stacking (````stack_batches_onto_model_device````) the resulting super batch, which comprises of batches from all memory mapped Nvidia GPUs (illustrated in the figure above).

And ````run_demo.py```` has minor changes; a list of ````torch.utils.data.DataLoader````s, one for each Nvidia GPU, is memory mapped, and the in-parallel enumeration and stacking tools allow us to scale ````tensordict_packages````'s speed advantage linearly with GPUs.

This provides an avenue through which PyTorch models can be trained on larger datasets while still benefiting from memory mapped tensors, and the simple example I've shown in ````multi_gpus_multi_batches_version```` offers directions for which AI Python engineers can uncover new and better ways to innovate PyTorch model training like we did here with ````TensorDict.MemoryMappedTensor````s--maxing out Nvidia GPUs' resources to gain the speed advantage scaled at larger datasets.

We should strive to make use of all of the Nvidia GPUs' resources available to us.






