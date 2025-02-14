# GMO GPU Cloud Example

Example scripts of fine-tuning LLMs on the [GMO GPU Cloud](https://gpucloud.gmo/) Slurm Cluster.

## Requirements

```bash
cd $HOME && git clone https://github.com/owner203/gmo-gpucloud-example.git

cd $HOME/gmo-gpucloud-example

# Initialize virtual environment
sbatch setup_env.sbatch
```

## Working Directory

`$HOME/gmo-gpucloud-example`

## LLMs and Datasets

- [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [alpaca_ja](https://github.com/shi3z/alpaca_ja)

```bash
cd $HOME/gmo-gpucloud-example

# Activate virtual environment
source scripts/activate_env.sh

mkdir -p LLM-Research/dataset

# Download Llama-3.1-8B-Instruct model
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir LLM-Research/Meta-Llama-3.1-8B-Instruct

# Download alpaca_ja dataset
curl -L -o LLM-Research/dataset/alpaca_cleaned_ja.json https://raw.githubusercontent.com/shi3z/alpaca_ja/refs/heads/main/alpaca_cleaned_ja.json
```

## Fine-Tuning

```bash
sbatch -p part-share --nodes=2 multi_node_sft.sbatch
```
