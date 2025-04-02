# veRL-multiturn-rollout Dev Release

## 环境配置

### 更新 verl-multiturn-rollout 的代码

```bash
# 在 docker 外面去 git pull 这个 repo，不要在 docker 里面
#否则你得先获得这个 repo 的权限并且把你的 docker 的 key 加到你的 github 账号里面
# 才可以直接 pull，但是在 docker 外面，atlas 上用的是我的号在鉴权，可以直接 pull
cd /.cache/veRL-multiturn-rollout
git pull
```

### 创建新的 docker

```bash
# 如果你的系统没有配置过 HF_TOKEN 和 WANDB_API_KEY，请先配置好
# 这里的 cache 映射路径是在 atlas 集群上，如果需要使用自己的路径，请自行修改
docker run -it --name h100_verl_{your_name} --gpus all \
    --shm-size 32g \
    -v /.cache:/root/.cache \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env "WANDB_API_KEY=$WANDB_API_KEY" \
    --ipc=host \
    lmsysorg/sglang:latest \
    /bin/bash
```

以后每次从 docker 里面 `exit` 出来，再用这个指令可以重启：

```bash
docker start -i h100_{your_name}
```

### 更新 python

```bash
mkdir -p /tmp
chmod 1777 /tmp
apt update
apt install -y python3.10 python3.10-venv
python3 -m ensurepip --upgrade
```

### 使用虚拟环境

```bash
# 创建虚拟环境
python3 -m venv ~/.python/veRL-multiturn-rollout

# 激活虚拟环境
source ~/.python/veRL-multiturn-rollout/bin/activate

# 安装 uv
python3 -m pip install uv
```

### clone veRL-multiturn-rollout 并切换到 async-tp 分支

PS：这个分支还是 private 的，需要先我们内部邀请加入，不然就下载下源代码，直接传到 docker 里面。我其实自己在 atals 的 `/.cache` 里面偷偷塞了一个，所以可以直接用：

```bash
cd ~
cp -r /root/.cache/veRL-multiturn-rollout .
cd veRL-multiturn-rollout
git checkout async-tp
```

### 配置 python 环境

先安装 veRL：

```bash
python3 -m uv pip install .
python3 -m uv pip install -r ./requirements.txt
```

安装 SGLang：

```bash
python3 -m uv pip install "sglang[all] @ git+https://github.com/sgl-project/sglang.git/@main#egg=sglang&subdirectory=python" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
```

手动安装 flash-attn：

```bash
python3 -m uv pip install wheel
python3 -m uv pip install packaging
python3 -m uv pip install flash-attn --no-build-isolation --no-deps
```

## 4 卡测试 SGLang

### 使用前需要配置好 `WANDB_API_KEY`

可以参考[这个过程](https://community.wandb.ai/t/where-can-i-find-the-api-token-for-my-project/7914)。

```bash
export WANDB_API_KEY={YOUR_WANDB_API_KEY}

# 定义这个时间戳函数
function now() {
    date '+%Y-%m-%d-%H-%M'
}
```

### 运行

```bash
# First make sure the now() function is available in current shell
# Create logs directory if it doesn't exist
mkdir -p logs

# Set GPUs and run with better log organization
export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python3 -m verl.trainer.main_ppo trainer.experiment_name=qwen7b_sft2_$(now) > logs/qwen7b_sft2_$(now).log 2>&1 &
```