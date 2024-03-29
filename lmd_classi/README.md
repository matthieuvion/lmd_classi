<div align="center">
  <h2>Notebooks: from LLM fine-tuning to quantized ONNX </h2>
  <p align="center">
    <a href="https://www.kaggle.com/amadevs/code"><img src="https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=Kaggle&logoColor=white" alt="Kaggle Badge"/></a>
    <a href="https://huggingface.co/gentilrenard"><img src="https://img.shields.io/badge/HuggingFace-black?style=flat&logo=huggingface&logoColor=white" alt="HuggingFace Badge"/></a>
  </p>
</div>
<br/>

Notebooks also available on kaggle. They should work on Google Colab too.

| Notebook | Description | Resource |
|--------------------------------------|-----------------------------------------------------------------------------|----------|
| lmd_setfit_modeling_logistic_head | Baseline model - few shots clf using SetFit | [notebook](https://www.kaggle.com/code/amadevs/lmd-setfit-modeling-logistic-head) |
| lmd_mistral_synthetic_gen_testprompt | Synthetic data gen - prepare dataset - prompts tests Mistral-7B-OpenHermes | [notebook](https://www.kaggle.com/code/amadevs/lmd-mistral-synthetic-gen-testprompt) |
| lmd_mistral_synthetic_gen_run        | Synthetic data gen - run (output : 2k synthetic samples) | [notebook](https://www.kaggle.com/code/amadevs/lmd-mistral-synthetic-gen-run) |
| lmd_mistral_synthetic_fine_tune      | Fine-tuning Mistral-7B-base for classi. (output: json label) w/ synth. data, using Unsloth (Qlora), Alpaca template | [notebook](https://www.kaggle.com/code/amadevs/lmd-mistral-synthetic-fine-tune) |
| lmd_setfit_mistral_evaluation        | Benchmark SetFit / fine-tuned Mistral (several experiments) | [notebook](https://www.kaggle.com/code/amadevs/lmd-setfit-mistral-evaluation) |
| lmd_setfit_mistral_inference         | Augment original dataset - voting ensemble SetFit + f-tuned LLM to infer 20k unlabeled comments| [notebook](https://www.kaggle.com/code/amadevs/lmd-setfit-mistral-inference) |
| lmd_multi-e5_train                   | multi-e5/bge embeddings + nn classifier on augmented dataset (several experiments) | [notebook](https://www.kaggle.com/code/amadevs/lmd-multi-e5-train) |
| e5_onnx_optimization                 | multi-e5 - ONNX conversion & optimization/quantization | [notebook](https://www.kaggle.com/code/amadevs/e5-onnx-optimization) |
| lmd_e5_evaluation                    | Benchmark all models - focus on global, minority class and inference latency | [notebook](https://www.kaggle.com/code/amadevs/lmd-e5-evaluation)
