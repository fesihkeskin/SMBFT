### **1. Data Splitting and Preprocessing**
These hyperparameters control how the dataset is split into training, validation, and test sets, as well as how the data is processed.

- **`train_ratio`**:  
  - **Purpose**: Specifies the proportion of the dataset to be used for training.  
  - **Effect**: A higher value means more data for training, which can improve model performance but may reduce the size of the validation and test sets, potentially leading to less reliable evaluation metrics.

- **`val_ratio`**:  
  - **Purpose**: Specifies the proportion of the dataset to be used for validation (out of the remaining data after the training split).  
  - **Effect**: A higher value provides more data for validation, which can improve the reliability of early stopping and hyperparameter tuning but reduces the size of the test set.

- **`patch_size`**:  
  - **Purpose**: Defines the size of the spatial patch extracted from the hyperspectral cube for training.  
  - **Effect**: Larger patches capture more spatial context, which can improve model performance for tasks requiring spatial information. However, larger patches increase computational cost and memory usage.

- **`stride`**:  
  - **Purpose**: Sets the step size for sliding the patch window over the image.  
  - **Effect**: Smaller strides (e.g., 1) result in more overlapping patches and more training samples, potentially improving accuracy but increasing computation and memory usage. Larger strides reduce overlap and speed up training, but may miss fine spatial details.

---

### **2. Training Control**
These hyperparameters govern the training process, including batch size, number of epochs, and early stopping.

- **`batch_size`**:  
  - **Purpose**: Specifies the number of samples processed together in one forward/backward pass.  
  - **Effect**: Larger batch sizes can speed up training but require more memory. Smaller batch sizes may lead to noisier gradient updates but can generalize better in some cases.

- **`epochs`**:  
  - **Purpose**: The maximum number of times the entire training dataset is passed through the model.  
  - **Effect**: More epochs allow the model to learn better but increase the risk of overfitting if the model trains for too long.

- **`early_stop`**:  
  - **Purpose**: Stops training early if the validation performance does not improve for a specified number of epochs.  
  - **Effect**: Prevents overfitting and saves computational resources by halting training when further improvement is unlikely.

---

### **3. Optimization**
These hyperparameters control the optimization process, including learning rate, weight decay, and dropout.

- **`lr` (Learning Rate)**:  
  - **Purpose**: Determines the step size for updating model weights during training.  
  - **Effect**: A higher learning rate speeds up training but may cause the model to converge to a suboptimal solution or diverge. A lower learning rate ensures stable convergence but may slow down training.

- **`weight_decay`**:  
  - **Purpose**: Adds a penalty to large weights in the loss function to prevent overfitting.  
  - **Effect**: Higher values encourage smaller weights, which can improve generalization but may underfit the data if set too high.

- **`dropout`**:  
  - **Purpose**: Randomly sets a fraction of neurons to zero during training to prevent overfitting.  
  - **Effect**: Higher dropout rates improve regularization but may reduce model capacity if set too high.

---

### **4. Model Architecture**
These hyperparameters define the structure of the Graph Attention Spectral Transformer (GAST) model and its variants.

- **`model_type`**:  
  - **Purpose**: Selects the model backend (e.g., `"gast"`, `"cnn"`, `"vit"`).  
  - **Effect**: Allows comparison of different architectures for ablation or benchmarking.

- **`embed_dim`**:  
  - **Purpose**: The dimensionality of the spectral embedding space.  
  - **Effect**: Higher values allow the model to capture more spectral features but increase computational cost and risk of overfitting.

- **`gat_hidden_dim`**:  
  - **Purpose**: The dimensionality of the hidden layers in the Graph Attention Network (GAT).  
  - **Effect**: Higher values increase the model's capacity to learn complex relationships but also increase computational cost and risk of overfitting.

- **`gat_heads`**:  
  - **Purpose**: The number of attention heads in the GAT layers.  
  - **Effect**: More heads allow the model to capture multiple types of relationships in the graph but increase computational cost.

- **`gat_depth`**:  
  - **Purpose**: The number of GAT layers in the model.  
  - **Effect**: More layers allow the model to learn deeper relationships but may lead to overfitting or vanishing gradients if set too high.

- **`transformer_heads`**:  
  - **Purpose**: Number of attention heads in the transformer encoder.  
  - **Effect**: More heads can improve the model's ability to capture complex dependencies but increase computation.

- **`transformer_layers`**:  
  - **Purpose**: Number of transformer encoder layers.  
  - **Effect**: More layers can increase model expressiveness but may also increase risk of overfitting and computational cost.

---

### **5. Ablation and Fusion Options**
These arguments enable ablation studies and control how spectral and spatial features are fused.

- **`disable_spectral`**:  
  - **Purpose**: If set, disables the spectral (MLP) branch for ablation.  
  - **Effect**: Allows assessment of the spatial branch's standalone contribution.

- **`disable_spatial`**:  
  - **Purpose**: If set, disables the spatial (GAT) branch for ablation.  
  - **Effect**: Allows assessment of the spectral branch's standalone contribution.

- **`fusion_mode`**:  
  - **Purpose**: Specifies the fusion strategy for combining spectral and spatial features.  
    - `"gate"`: Learnable gated fusion (default, recommended)  
    - `"concat"`: Concatenation of features  
    - `"spatial_only"`: Use only spatial branch  
    - `"spectral_only"`: Use only spectral branch  
  - **Effect**: Enables systematic ablation and comparison of fusion strategies.

---

### **6. System Parameters**
These hyperparameters control system-level settings, such as random seeds and parallelism.

- **`seed`**:  
  - **Purpose**: Sets the random seed for reproducibility.  
  - **Effect**: Ensures that the same random operations (e.g., data splitting, weight initialization) produce identical results across runs.

- **`num_workers`**:  
  - **Purpose**: Specifies the number of worker threads for data loading.  
  - **Effect**: Higher values speed up data loading but may cause bottlenecks if the CPU is overloaded.

- **`output_dir`**:  
  - **Purpose**: Specifies the directory where model checkpoints and logs are saved.  
  - **Effect**: Ensures that training artifacts are stored for later use, such as evaluation or resuming training.

---

### **Summary of Effects**
- **Improving Model Performance**: Parameters like `patch_size`, `stride`, `embed_dim`, `gat_hidden_dim`, `gat_heads`, and `transformer_layers` directly affect the model's ability to learn and generalize.
- **Preventing Overfitting**: Parameters like `dropout`, `weight_decay`, and `early_stop` help regularize the model.
- **Speed vs. Accuracy Trade-off**: Parameters like `batch_size`, `epochs`, `lr`, and `stride` balance training speed and model accuracy.
- **Ablation and Fusion**: Options like `disable_spectral`, `disable_spatial`, and `fusion_mode` enable systematic analysis of model components.
- **Reproducibility and Efficiency**: Parameters like `seed` and `num_workers` ensure consistent results and efficient resource utilization.