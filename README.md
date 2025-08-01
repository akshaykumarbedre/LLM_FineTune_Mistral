# LLM Fine-Tuning with Mistral-7B and TorchTune

This repository demonstrates **instruction fine-tuning of the Mistral-7B model** using **TorchTune** framework, optimized for **AMD MI300 GPU** environments. The project focuses on creating conversational AI assistants using OpenAssistant-style English datasets.

## 🎯 Project Overview

The goal of this project is to fine-tune large language models (specifically Mistral-7B) to improve their conversational abilities using instruction-following datasets. We utilize the OpenAssistant dataset, converting it to OpenAI chat format for better compatibility with modern training pipelines.

**Key Features:**
- ✅ Mistral-7B model fine-tuning using LoRA (Low-Rank Adaptation)
- ✅ OpenAssistant dataset preprocessing and format conversion
- ✅ TorchTune integration for efficient training
- ✅ AMD MI300 GPU optimization (also compatible with CUDA)
- ✅ Multiple fine-tuning workflows for different use cases

## 🔗 Model & Dataset Links

### Pre-trained Models
- **Fine-tuned Model**: [Akshaykumarbm/OpenAssisted-English-Mistral-7b](https://huggingface.co/Akshaykumarbm/OpenAssisted-English-Mistral-7b)
- **Base Model**: [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)

### Datasets
- **Processed Dataset**: [Akshaykumarbm/oasst-english-openai-formate](https://huggingface.co/datasets/Akshaykumarbm/oasst-english-openai-formate)
- **Source Dataset**: [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1)

## 🛠️ Environment Setup

### Prerequisites
- Python 3.8+
- CUDA 11.8+ or ROCm 5.4+ (for AMD MI300)
- 16GB+ GPU memory recommended
- Hugging Face account and token

### Installation Steps

1. **Clone the repository:**
```bash
git clone https://github.com/akshaykumarbedre/LLM_FineTune_Mistral.git
cd LLM_FineTune_Mistral
```

2. **Install TorchTune and dependencies:**
```bash
pip install torch torchvision
pip install torchtune
pip install transformers datasets accelerate
```

3. **For AMD MI300 setup:**
```bash
# Install ROCm PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.6

# Verify ROCm installation
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

4. **Set up Hugging Face authentication:**
```bash
pip install huggingface_hub
huggingface-cli login
```

## 📚 Notebook Documentation

This repository contains three main workflows, each with dedicated notebooks:

### 1. Complete Fine-tuning with Human-like Conversations

**Location**: `complete_fine_tuing_human_like_convestion/`

#### `dataprepocesing.ipynb`
**Purpose**: Converts OpenAssistant dataset to OpenAI chat format for fine-tuning.

**Workflow**:
1. **Load OpenAssistant Dataset**: Downloads the raw OASST1 dataset
2. **Filter English Conversations**: Extracts only English language conversations
3. **Build Conversation Trees**: Reconstructs conversation threads from individual messages
4. **Convert to OpenAI Format**: Transforms data into `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}` format
5. **Upload to Hugging Face**: Saves processed dataset for reuse

**Key Output**: Creates training and validation datasets in OpenAI format.

#### `Validation.ipynb`
**Purpose**: Tests the fine-tuned model performance and creates validation datasets.

**Workflow**:
1. **Load Validation Data**: Uses validation split from processed dataset
2. **Model Testing**: Loads the fine-tuned Mistral-7B model
3. **Generate Responses**: Tests model on validation questions
4. **Compare Outputs**: Shows original answers vs. model-generated responses
5. **Performance Analysis**: Basic comparison of response quality

### 2. Fine-tuning with Custom Datasets

**Location**: `fine_tuing_custem_dataset/`

#### `Data_creating_expro.ipynb`
**Purpose**: Demonstrates model download and preparation using TorchTune.

**Workflow**:
1. **Download Base Models**: Uses `tune download` to get Mistral-7B and Gemma-2B models
2. **Model Storage**: Saves models to `/tmp/` directory for training
3. **Configuration Setup**: Prepares configuration files for different models
4. **Recipe Exploration**: Shows available TorchTune recipes for fine-tuning

#### `fineTuing.ipynb`
**Purpose**: Executes the actual fine-tuning process using TorchTune.

**Workflow**:
1. **Recipe Selection**: Chooses appropriate TorchTune recipe (LoRA single device)
2. **Configuration**: Sets up training parameters, learning rates, batch sizes
3. **Training Execution**: Runs the fine-tuning process
4. **Checkpoint Management**: Saves and manages model checkpoints
5. **Model Export**: Converts trained model for deployment

**Configuration Files**:
- `gemma2bconfig.yaml`: LoRA fine-tuning config for Gemma-2B
- Settings: rank=64, alpha=128, lr=2e-5, batch_size=8

### 3. Fine-tuning with Pre-built Datasets

**Location**: `Fine_tuning_pre_buil_dataset/`

#### `Gemma2B_finetuing.ipynb`
**Purpose**: Demonstrates fine-tuning workflow using pre-built datasets and recipes.

**Workflow**:
1. **Environment Setup**: Installs TorchTune and dependencies
2. **Recipe Exploration**: Lists all available TorchTune recipes
3. **Model Download**: Downloads pre-trained models
4. **Configuration**: Uses standard TorchTune configurations
5. **Training**: Executes fine-tuning with built-in datasets (Alpaca, etc.)

**Configuration Files**:
- `config.yaml`: Standard LoRA config for Llama3.1
- `custom_config.yaml`, `custom_eval_config.yaml`: Custom training configurations

## 🚀 How to Run the Notebooks

### Option 1: Complete Workflow (Recommended for beginners)

1. **Start with Data Preprocessing:**
```bash
cd complete_fine_tuing_human_like_convestion/
jupyter notebook dataprepocesing.ipynb
```
- Update your Hugging Face token in the notebook
- Run all cells to process the OpenAssistant dataset
- This creates the training dataset

2. **Run Fine-tuning:**
```bash
cd ../fine_tuing_custem_dataset/
jupyter notebook fineTuing.ipynb
```
- Follow the notebook to download models and run training
- Training time: ~2-4 hours on MI300/A100

3. **Validate Results:**
```bash
cd ../complete_fine_tuing_human_like_convestion/
jupyter notebook Validation.ipynb
```
- Test your fine-tuned model
- Compare outputs with expected responses

### Option 2: Pre-built Dataset Workflow

```bash
cd Fine_tuning_pre_buil_dataset/
jupyter notebook Gemma2B_finetuing.ipynb
```
- Uses built-in Alpaca dataset
- Faster setup, good for experimentation

### Option 3: Command Line Training

For production training, use TorchTune CLI:

```bash
# Download model
tune download mistralai/Mistral-7B-v0.1 --hf-token YOUR_TOKEN

# Run training
tune run lora_finetune_single_device --config mistral/7B_lora_single_device

# Custom config
tune run lora_finetune_single_device --config path/to/your/config.yaml
```

## ⚙️ Training Configuration

### Key Parameters
- **Model**: Mistral-7B-v0.1
- **Method**: LoRA (Low-Rank Adaptation)
- **LoRA Rank**: 8-64 (higher = more parameters, better quality)
- **Learning Rate**: 2e-5 to 3e-4
- **Batch Size**: 2-8 (adjust for GPU memory)
- **Gradient Accumulation**: 8 steps
- **Epochs**: 1-3 (more epochs risk overfitting)

### Memory Requirements
- **Mistral-7B LoRA**: ~12GB GPU memory
- **Gemma-2B LoRA**: ~6GB GPU memory
- **Full Fine-tuning**: 24GB+ GPU memory

## 📊 Current Status

### ✅ Completed Features
- [x] OpenAssistant dataset preprocessing
- [x] Mistral-7B LoRA fine-tuning
- [x] OpenAI format conversion
- [x] TorchTune integration
- [x] Model deployment to Hugging Face
- [x] AMD MI300 compatibility
- [x] Multiple training workflows

### ❌ Not Yet Implemented
- [ ] **Evaluation Pipeline**: No METEOR/ROUGE/BLEU scoring implemented
- [ ] **Automated Hyperparameter Tuning**: Manual configuration required
- [ ] **Multi-GPU Training**: Currently single-device only
- [ ] **Quantization**: No INT8/FP16 optimization
- [ ] **Custom Evaluation Metrics**: Basic comparison only

## 🔮 Future Work

### Planned Improvements
1. **Evaluation Pipeline**
   - Implement METEOR, ROUGE, BLEU metrics
   - Add perplexity and human evaluation scores
   - Create automated evaluation scripts

2. **Dataset Expansion**
   - Convert more datasets to OpenAI format
   - Support for multi-turn conversations
   - Custom domain-specific datasets

3. **Model Scaling**
   - Support for Mistral-8x7B (Mixture of Experts)
   - Llama-2/3 fine-tuning workflows
   - Code-specific model variants

4. **Performance Optimization**
   - Multi-GPU training support
   - Gradient accumulation optimization
   - Memory-efficient training techniques

5. **Deployment Features**
   - FastAPI inference server
   - Docker containerization
   - Cloud deployment scripts

### Contributing Areas
- **Data Engineering**: More dataset converters
- **Evaluation**: Better metrics and benchmarks
- **Optimization**: Training speed improvements
- **Documentation**: More examples and tutorials

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch_size in config
   - Increase gradient_accumulation_steps
   - Enable activation checkpointing

2. **AMD ROCm Setup Issues**
   - Ensure ROCm 5.4+ is installed
   - Use ROCm-specific PyTorch build
   - Check `rocm-smi` for GPU detection

3. **Hugging Face Authentication**
   - Run `huggingface-cli login`
   - Check token permissions
   - Verify internet connection

4. **Model Download Failures**
   - Check disk space (models are 13GB+)
   - Verify Hugging Face token
   - Use `--ignore-patterns` to skip large files

### Performance Tips
- Use `compile=True` in config for 10-20% speedup
- Enable `fused=True` in optimizer
- Set `dtype: bf16` for memory savings
- Use `packed=True` for datasets when possible

## 📝 License

This project is open-source and available under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- **TorchTune Team**: For the excellent fine-tuning framework
- **OpenAssistant**: For providing high-quality conversational datasets
- **Mistral AI**: For the powerful base language models
- **Hugging Face**: For model hosting and dataset infrastructure
- **AMD**: For MI300 GPU support and ROCm ecosystem

---

**Questions or Issues?** Open a GitHub issue or reach out to the community. Happy fine-tuning! 🚀