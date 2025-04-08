# CM-GAN for Renewable Energy Scenario Generation

## Project Purpose
We leverage a Cross-Modal Generative Adversarial Network (CM-GAN) to generate day-ahead renewable energy scenarios, specifically focusing on photovoltaic (PV) power generation. The model takes historical PV generation data and geospatial information as input (see solar.xlsx and gps.csv) and outputs PV generation scenarios.

## Model Architecture
The CM-GAN model is designed to capture both spatial and temporal dependencies in PV generation data. It consists of the following key components:

1. **Generator**:
   - Consists of two spatial-temporal transformer models(detailed below).
   - Captures spatial correlations between PV stations and temporal dynamics in power generation sequences.
   - Outputs PV power scenarios with shape `[batch_size, 69, 288, 1]`.

2. **Discriminator**:
   - Consists of two spatial-temporal transformer models and Flatten layer.
   - Utilizes adversarial training with WGAN-GP (Wasserstein GAN with Gradient Penalty) loss.

3. **Attention Mechanisms**:
   - **Spatial Self-Attention (`SpatialAttention`)**:
     - **Purpose**: Models spatial relationships between PV stations using geospatial information.
     - **Structure**:
       - **Embedding Layer**: 
         - Incorporates an adjacency matrix (`self.A`, shape `[input_dim, input_dim]`) , initialized with Xavier uniform distribution.
         - Combines this with a learnable spatial embedding matrix (`self.W_a_S`, shape `[input_dim, seq_len, hidden_dim]`), which embeds the adjacency matrix into a higher-dimensional space.
         - Computes `ES = AW_a_S`, yielding a spatial embedding tensor of shape `[input_dim, seq_len, hidden_dim]`, expanded to `[batch_size, input_dim, seq_len, hidden_dim]` for broadcasting.
       - **Query, Key, Value Projections**:
         - Linear layers (`self.W_q`, `self.W_k`, `self.W_v`) transform the input (`XS`) into query (`QS`), key (`KS`), and value (`VS`) tensors, each of shape `[batch_size, input_dim, seq_len, hidden_dim]`.
       - **Attention Computation**:
         - Calculates attention scores (`energy`) as `QS · KS^T / sqrt(hidden_dim)`, resulting in shape `[batch_size, input_dim, input_dim, hidden_dim]`.
         - Applies softmax along the `input_dim` dimension to obtain the attention matrix (`M^S`).
         - Computes context as `M^S · VS`, yielding `[batch_size, input_dim, seq_len, hidden_dim]`.
       - **Feed-Forward Network (FFN)**:
         - Three linear layers (`self.W_0`, `self.W_1`, `self.W_2`) with ReLU activations process the context, maintaining the shape `[batch_size, input_dim, seq_len, hidden_dim]`.
       - **Normalization**: Applies layer normalization (`self.norm`) with a residual connection to stabilize training.

   - **Temporal Self-Attention (`TemporalAttention`)**:
     - **Purpose**: Captures temporal dynamics across the 288 time steps of a day.
     - **Structure**:
       - **Positional Encoding**:
         - Uses a learnable positional encoding matrix (`self.pos_encoding`, shape `[seq_len, hidden_dim]`), initialized with Xavier uniform distribution.
         - Expands to `[batch_size, input_dim, seq_len, hidden_dim]` and adds to the input (`XT = hl + ET`) to encode temporal positions.
       - **Query, Key, Value Projections**:
         - Linear layers (`self.W_q`, `self.W_k`, `self.W_v`) project the input (`XT`) into query (`QT`), key (`KT`), and value (`VT`) tensors, each of shape `[batch_size, input_dim, seq_len, hidden_dim]`.
       - **Attention Computation**:
         - Computes attention scores (`energy`) as `QT · KT^T / sqrt(hidden_dim)` via `torch.einsum`, resulting in shape `[batch_size, input_dim, input_dim, hidden_dim]`.
         - Applies softmax along the `seq_len` dimension to produce the attention matrix (`M^T`).
         - Calculates context as `M^T · VT`, yielding `[batch_size, input_dim, seq_len, hidden_dim]`.
       - **Feed-Forward Network (FFN)**:
         - Three linear layers (`self.W_0`, `self.W_1`, `self.W_2`) with ReLU activations process the context, maintaining the shape `[batch_size, input_dim, seq_len, hidden_dim]`.
       - **Normalization**: Applies layer normalization (`self.norm`) with a residual connection for training stability.

## Input Data
The model processes two primary types of input data:

1. **PV Generation Data**:
   - **Format**: Time-series data with shape `[17280, 69]`, representing power output from 69 PV stations in 2 months.
   - **Temporal Granularity**: 5-minute intervals, yielding 288 time steps per day.
   - **Source**: `solar.xlsx` (loaded via `pandas` and normalized using `MinMaxScaler`).
   - **Purpose**: Provides historical PV power data for training and scenario generation.

2. **Geospatial Information**:
   - **Format**: GPS coordinates with shape `[69, 2]`, containing latitude and longitude for 69 PV stations.
   - **Source**: `gps.csv`.
   - **Purpose**: Used to compute an adjacency matrix via a Gaussian kernel (σ = 100), capturing spatial dependencies between stations.

## Output Results
The model generates diverse PV power scenarios with the following characteristics:
- **Shape**: `[batch_size, 69, 288, 1]`, where:
  - `batch_size`: Number of generated scenarios.
  - `69`: Number of PV stations.
  - `288`: Time steps per day (24 hours at 5-minute intervals).
  - `1`: Power output value (in MW) at each time step.
- **Visualization**: Outputs are compared with real data using plots (e.g., time-series curves and distributions) saved in the `samples` directory.

## Training Process
- **Data Loading**: Handled by `load_solar()` in `load.py`, splitting data into training and test sets (80-20 split).
- **Adversarial Training**: Implemented in `CMGANTrainer` (`train.py`) with:
  - Generator iterations (`ngen = 2`) and discriminator iterations (`ndisc = 3`) per step.
  - Learning rates: `g_lr = 0.0001`, `d_lr = 0.0001`.
  - Optimizer: Adam with `beta1 = 0.0`, `beta2 = 0.9`.
- **Loss Function**: WGAN-GP with a gradient penalty coefficient (`lambda_gp = 10`).
- **Total Steps**: Configurable (default: 100 epochs).
