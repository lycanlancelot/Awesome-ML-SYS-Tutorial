# GRPO

Group Relative Policy Optimization `grpo`,  a variant of Proximal Policy Optimization(PPO), enhances mathematical reasoning abilities while concurrently optimizing the memory usage of PPO. This doc will first introduce how GRPO works and then show how we add SGLang as an alternative inference backend for TRL, specifically the [GRPO Trainer](https://huggingface.co/docs/trl/main/en/grpo_trainer), which also functions in the [Open-R1](https://github.com/huggingface/open-r1) project.

# 1. How GRPO works

Compared with PPO, GRPO doesn’t have the **value/critic model** to estimate total value. The algorithm computes the normalized reward for each output to derive advantages and updates the **reward model** to enhance training performance.

这里说的 update，是指参数更新么？应该不是吧，所以这里的 update 是什么意思？

GRPO is composed of four steps:

- Generating completions
- Computing the advantage
- Estimating the KL divergence
- Computing the loss

<div align="center">
<img src="GRPO_Images/grpo-main.png" width="100%">
</div>


## 1.1 Generating completions

At each training step, we sample a batch of prompts and generate a set of $G$ completions for each prompt (denoted as $o_{i = 1, 2, ..., G}$).

## 1.2 Computing the advantage

For each of the $G$ sequences, we compute the reward using a **reward model**. To align with the comparative nature of reward models—typically trained on datasets of comparisons between outputs for the same question—the advantage is calculated to reflect these relative comparisons. It is normalized as follows:

$$\hat{A}_{i,t} = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$$

This approach gives the method its name: **Group Relative Policy Optimization (GRPO)**, since it uses the relative reward to compute the advantage.

Hugging Face uses the above equation to compute advantages. In [GRPO paper](https://arxiv.org/abs/2402.03300), the author named it Outcome Supervision RL with GRPO. The author also found another method named Process Supervision RL with GRPO.

什么是 Supervision RL？什么是 Process Supervision RL？

We can also leverage the information in each step. Formally, given the question *q* and *G* sampled outputs {$o_1$, $o_2$, … , $o_G$}, a process reward model is R = {{$r_1^{index(1)}$, …, $r_1^{index(K_1)}$}, … , {$r_G^{index(1)}$, …, $r_G^{index(K_G)}$}}, where  $index(j)$ is the end token index of $j$-th step.

Normalize each reward:

$$
\tilde{r}_i^{index(j)} = \frac{r_i^{index(j)} - mean(R)}{std(R)}
$$

The advantages of each output in each step:

$$
\hat{A}_{i,t} = \sum_{index(j)\geq t}\tilde r_i^{index(j)}
$$

所以说 advantage 可以是 step wised 的，也可以是 output wised 的么？

## 1.3 Estimating the KL divergence

KL divergence is estimated using the approximator, which is defined as follows:

$$
\mathrm{D}_{\mathrm{KL}}[\pi_{\theta}||\pi_{\mathrm{ref}}] = \frac{\pi_{\mathrm{ref}}(O_{i,t} \mid q, O_{i,<t})}{\pi_{\theta}(O_{i,t} \mid q, O_{i,<t})} - \log \frac{\pi_{\mathrm{ref}}(O_{i,t} \mid q, O_{i,<t})}{\pi_{\theta}(O_{i,t} \mid q, O_{i,<t})} - 1
$$
r
## 1.4 Computing the loss

The objective is to **maximize the advantage** while ensuring that the model **remains close to the reference policy**. Consequently, the loss is defined as follows:

The objective is to maximize the advantage while ensuring that the model remains close to the reference policy. Consequently, the loss is defined as follows:

$$
\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left[ \frac{\pi_{\theta}(O_{i,t} \mid q, O_{i,<t})}{\pi_{\text{ref}}(O_{i,t} \mid q, O_{i,<t})} \right]_{\text{no_grad}} \hat{A}_{i,t} - \beta \mathrm{D}_{\mathrm{KL}}[\pi_{\theta}||\pi_{\mathrm{ref}}],
$$

- the first term represents the scaled advantage
- the second term penalizes deviations from the reference policy through KL divergence.

In the original paper, this formulation is generalized to account for multiple updates after each generation by leveraging the **clipped surrogate objective**:

![Screenshot 2025-02-18 at 12.50.52 AM.png](GRPO_Images/GRPO_Loss.png)

where clip(⋅,1−*ϵ*,1+*ϵ*) ensures that updates stay close to the reference policy by keeping the policy ratio between 1−*ϵ* and 1+*ϵ*. However, since TRL follows the original paper in performing only one update per generation, we can simplify the loss to the first form.

## **1.5 Logged metrics**

The GRPO Trainer logs the following metrics:

- `completion_length`: The average completion length.
- `reward/{reward_func_name}`: The reward computed by each reward function.
- `reward`: The average reward.
- `reward_std` : The average standard deviation within reward groups.
- `kl` : The average KL divergence between the model and the reference model calculated on completions.

# 2. Customized GRPO

## 2.1 Feature and Functions

[Speed up training with vLLM](https://huggingface.co/docs/trl/main/en/speeding_up_training?vllm+examples=GRPO#vllm-for-fast-generation-in-online-methods)

![Screenshot 2025-02-17 at 11.13.02 PM.png](GRPO_Images/Usage.png)

![Screenshot 2025-02-17 at 11.14.39 PM.png](GRPO_Images/Parameters.png)

## 2.2 How vLLM is used in [grpo_trainer.py](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py):

### **2.2.1. Conditional Setup and Import**

- **Import and Availability Check:**

At the top of the file, the code conditionally imports the vLLM modules:

```python
if is_vllm_available():
    from vllm import LLM, SamplingParams
```

This ensures that vLLM’s generation engine is available only when installed.

- **Flag in the Trainer:**

In the trainer’s constructor `__init__`, the flag self.use_vllm is set from the configuration (`args.use_vllm`). This flag governs whether to use vLLM for text generation.

### **2.2.2. Initialization and Dedicated GPU Selection**

- **Dedicated Device Assignment:**

When `use_vllm` is True, the trainer (but only on the main process) determines a dedicated GPU for generation. For example:

```python
if self.accelerator.is_main_process:
    vllm_device = self.args.vllm_device
    if vllm_device == "auto":
        if torch.cuda.device_count() == 1:
            vllm_device = "cuda:0"
        else:
            vllm_device = f"cuda:{self.accelerator.num_processes}"
```

This logic assigns one GPU (or a GPU outside of those used for training) exclusively for the vLLM generation task, thereby offloading generation work from the training GPUs.

- **Patching for Compatibility:**

Since vLLM isn’t inherently designed to work with the distributed setup from Accelerate, the code applies two patches:

```python
world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
profiling_patch = patch("vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None)
with world_size_patch, profiling_patch:
    self.llm = LLM(
        model=model.name_or_path,
        device=vllm_device,
        gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
        dtype=self.args.vllm_dtype,
        enable_prefix_caching=True,
        max_model_len=self.args.vllm_max_model_len,
    )
self.sampling_params = SamplingParams(
    temperature=args.temperature,
    max_tokens=self.max_completion_length,
)
```

These patches make sure that:

- The vLLM engine sees a **“world size”** of 1 (since it’s running on a dedicated device).
- Certain profiling checks that aren’t applicable in this setting are bypassed.
- **Synchronization:**

After setting up vLLM, the main process calls:

```python
self.accelerator.wait_for_everyone()
```

When using vLLM, the main process is responsible for loading the model weights. This can cause process desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, this function ensures **all processes are synchronized** after the dedicated generation device is set up.

### **2.2.3. Using vLLM for Generation During Training**

- **Moving Weights to vLLM:**

Before generating completions, the trainer calls **`_move_model_to_vllm(self)`**. This method extracts the model’s state (merging adapters if needed) and loads the weights into the vLLM engine’s internal model:

```python
if self.accelerator.is_main_process:
    llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())
```

<aside>
💡

This transfer is key because vLLM is responsible for **generating text**, and its internal model must reflect **the latest training weights**.

</aside>

- **Generation Workflow:**

In the `_prepare_inputs` method, when using vLLM:

- The main process gathers **all prompt texts** across GPUs.
- It then **generates completions** with:

```python
outputs = self.llm.generate(all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False)
```

- The completions are **extracted**, and the results are **broadcast** to all processes so that every GPU gets its corresponding slice of generated text.
- **Offloading and Efficiency:**

By offloading the generation task to a dedicated GPU (via vLLM), the training GPUs remain fully occupied with gradient computations, thus speeding up training throughput.

Below is the revised section that replaces the vLLM‐specific code with a SGLang‐based solution, taking into account that SGLang runs as an external server and does not provide in‐process engine classes.

This document describes the modifications we made to support SGLang as an inference backend alongside vLLM in the GRPOTrainer. Unlike vLLM—which runs as an in-process engine using dedicated classes (e.g. LLM, SamplingParams) and requires patching of distributed methods—SGLang is deployed as a standalone server with HTTP endpoints (compatible with OpenAI’s APIs). As a result, our integration leverages HTTP requests to manage weight updates and generate completions.

## **2.3 Replacing vLLM with SGLang**

To substitute **vLLM** with **SGLang**, we must account for the differences in API and internal architecture. The following steps outline the necessary modifications:

### **2.3.1 Import and Availability Check**

- **Configuration Flag:**

Instead of importing in‑process engine classes for SGLang, we introduce a configuration flag (`use_sglang`) in our arguments (e.g., in GRPOConfig). This flag signals that generation should be offloaded to a SGLang server. Since SGLang is accessed via HTTP calls, there’s no need to import objects like SGLangEngine.

- **No In‑Process Imports for Generation:**

Instead, we ensure that the SGLang server is reachable (or launch it within our code) and then use helper utilities (from `sglang.utils`) to manage the server lifecycle.

### **2.3.2 Initialization on a Dedicated GPU**

- **Server Launch and Device Assignment:**

Rather than creating an in‑process generation engine (as with vLLM), we launch the SGLang server as an external process on a dedicated GPU. For example, in the trainer’s `__init__`, we added:

```python
if self.args.use_sglang:
    # Assign a dedicated GPU for the SGLang server (e.g., "cuda:1")
    sglang_gpu_id = self.args.sglang_device.split(':')[-1]
    sglang_command = (
        f"CUDA_VISIBLE_DEVICES={sglang_gpu_id} python -m sglang.launch_server "
        f"--model-path {model_id} --host 0.0.0.0"
    )
    from sglang.utils import launch_server_cmd, wait_for_server
    self.server_process, port = launch_server_cmd(sglang_command)
    wait_for_server(f"http://localhost:{port}")
    self.sglang_server_url = f"http://localhost:{port}"
```

This command dedicates one GPU exclusively for the SGLang server, which will handle all generation requests.

- **Process Synchronization:**

After launching the server, we call:

```python
self.accelerator.wait_for_everyone()
```

to ensure that all distributed processes are synchronized before proceeding.

### **2.3.3 Generation and Weight Synchronization**

- **Weight Synchronization:**

With vLLM, we update weights in‑process via a helper like `_move_model_to_vllm()`. For SGLang, weight updates occur externally. We implement a helper function `_update_sglang_weights()` that calls SGLang’s `**/update_weights_from_disk**` API to refresh the server’s model state:

We revised `_update_sglang_weights` to robustly update model weights on the SGLang server by calling its /update_weights_from_disk API. This function now:
- Checks if the checkpoint exists.
- Sends an HTTP POST with a timeout.
- Checks for a success flag in the response.
- Optionally flushes the cache after the update.

- **About SGLang update weights**

    **We meet some issue when trying to revise this function. To fix this, we must choose one of two paths:**

    1.	**Initialize the weight update group on the SGLang server** so that it supports distributed updates (and then continue using /update_weights_from_distributed). This means modifying the server’s initialization (in ModelRunner) to call its init_weights_update_group function.

    2.	**Add a checkpointing mechanism in the training loop and use the disk-based update endpoint** `/update_weights_from_disk` (which doesn’t require a weight update group). For this, update GRPOConfig to include a checkpoint_path and ensure that a checkpoint is written before calling the update.


In our current workflow we load the model directly from Hugging Face – which means we never had a “checkpoint” per se. To use the SGLang `/update_weights_from_disk` endpoint (our “second choice”), we need to supply a checkpoint file. One straightforward solution is to add a new field (say, checkpoint_path) to our `GRPOConfig` and then, before training starts, save the current model weights to that location. Then, when `_update_sglang_weights()` is called, it will have a valid file from which the SGLang server can reload the weights.

```python
def _update_sglang_weights(self):
    """
    Update the model weights on the SGLang server via its API.
    This function assumes that the training loop writes the latest checkpoint to self.args.checkpoint_path.
    It performs additional checks and error handling to ensure the server successfully updates its weights.
    """
    import os
    import requests

    checkpoint = self.args.checkpoint_path
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint path {checkpoint} does not exist.")

    payload = {"model_path": checkpoint}
    try:
        response = requests.post(
            f"{self.sglang_server_url}/update_weights_from_disk",
            json=payload,
            timeout=60,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"Weight update request failed: {e}")

    res_json = response.json()
    if not res_json.get("success", False):
        raise RuntimeError(f"Failed to update weights on SGLang server: {res_json.get('message', 'No message provided')}")

    # Optionally flush the cache after updating weights
    try:
        flush_response = requests.post(f"{self.sglang_server_url}/flush_cache", timeout=30)
        if not flush_response.json().get("success", True):
            print(f"Warning: Cache flush failed: {flush_response.json().get('message', 'No message provided')}")
    except requests.RequestException as e:
        print(f"Warning: Cache flush request failed: {e}")

    print(f"SGLang weights updated successfully: {res_json.get('message')}")
```

This function is called whenever the training step advances (e.g., if global_step changes).

- **Generation Call:**

In the `_prepare_inputs()` method, we replace the in‑process generation call with an HTTP request to SGLang’s `/generate` endpoint:

> - This branch mirrors the vLLM branch in structure but uses HTTP requests instead of an in‑process generation call.
- It maintains consistency with the default postprocessing (padding, slicing, and concatenation) so that the rest of the training pipeline remains unchanged.
>

```python
if self.use_sglang:
    # Update weights if the training step has advanced.
    if self.state.global_step != self._last_loaded_step:
        self._update_sglang_weights()
        self._last_loaded_step = self.state.global_step

    # Gather all prompt texts from all processes.
    all_prompts_text = gather_object(prompts_text)
    if self.accelerator.is_main_process:
        import requests
        payload = {
            "text": all_prompts_text,
            "sampling_params": self.sglang_sampling_params,
        }
        response = requests.post(f"{self.sglang_server_url}/generate", json=payload)
        generated_texts = response.json().get("text", [])
        completion_ids = [self.processing_class.encode(text) for text in generated_texts]
    else:
        completion_ids = [None] * len(all_prompts_text)

    # Broadcast and slice the generated completions.
    completion_ids = broadcast_object_list(completion_ids, from_process=0)
    process_slice = slice(
        self.accelerator.process_index * len(prompts),
        (self.accelerator.process_index + 1) * len(prompts),
    )
    completion_ids = completion_ids[process_slice]
    completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
    completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
    prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
```

- **Broadcasting Results:**

We then broadcast the generated completions from the main process across all processes using `broadcast_object_list()` and slice the results according to each process’s index.

### **2.3.4 API and Integration Adjustments**

- **Parameter Differences:**

Since SGLang’s API mimics OpenAI’s endpoints, we pass sampling parameters as a **JSON payload** (e.g., `temperature`, `max_new_tokens`) to the `/generate` endpoint.

- **Error Handling and Device Checks:**

We update error messages and checks for SGLang without needing to patch distributed functions—since SGLang runs as an external service, it operates independently of the training environment.

### **2.3.5 Overall Code Changes Recap**

- **Backend Flags:**

Added use_sglang (and retained use_vllm) in configuration to let users choose the inference backend.

- **Server Initialization:**

In the `__init__` method, if `use_sglang` is True, launch the SGLang server on a dedicated GPU and set `self.sglang_server_url`.

- **Weight Updates:**

Implemented a robust **`_update_sglang_weights()`** function that ensures the SGLang server updates its model weights from the latest checkpoint, with error handling and cache flushing.

- **Generation Branch:**

Modified **`_prepare_inputs()`** to branch based on the selected backend:

- **SGLang Branch:**

Uses HTTP calls to SGLang’s `/generate` endpoint, then converts returned texts to token IDs, broadcasts, and postprocesses.

### **2.3.6 Testing and Next Steps**

1. **Test the SGLang Server:**

Ensure that the SGLang server launches correctly on the dedicated GPU and that its `/generate` and `/update_weights_from_disk` endpoints respond as expected.

1. **Integration Test:**

Run the modified GRPOTrainer on a small dataset and verify that:

- Weights are updated on the SGLang server when the training step advances.
- Generation results are correctly broadcast and postprocessed.
- The overall training loop runs without errors.
1. **Cleanup:**
- Consider adding a cleanup method (or a destructor) in GRPOTrainer to properly shut down the SGLang server process when training finishes.
- Add SGLang Installation and Support doc in the readme.md

## **2.4 Summary**

### **2.4.1 vLLM’s Role:**

vLLM offloads text generation to an in-process engine running on a dedicated GPU by patching distributed training functions, moving weights directly, and calling an internal generate() method.

### **2.4.2 Replacing vLLM with SGLang:**

To swap vLLM with SGLang, we would:

1. Introduce a configuration flag (e.g. use_sglang) and assign a dedicated GPU for the SGLang server.
2. Launch the SGLang server externally using a command (with appropriate GPU assignment) and wait for it to initialize.
3. Replace the in-process generation call in _prepare_inputs() with an HTTP POST to the SGLang /generate endpoint.
4. Implement a helper to update model weights on the SGLang server via its /update_weights_from_disk API.
5. Adjust parameter names, error handling, and synchronization logic to match SGLang’s external server API.

# **References:**

- SGLang Documentation – [https://docs.sglang.ai/backend/](https://docs.sglang.ai/backend/)
- HuggingFace GRPO Trainer Documentation - [https://huggingface.co/docs/trl/main/en/grpo_trainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- Speeding Up Training - https://huggingface.co/docs/trl/main/en/speeding_up_training?vllm+examples=GRPO#vllm-for-fast-generation-in-online-methods
- GRPO Trainer in TRL - https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py