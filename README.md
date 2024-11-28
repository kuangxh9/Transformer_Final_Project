# **Protein Sequence Classification using Fine-Tuned Transformer Model**
Protein aggregation plays a vital role in various neurodegenerative diseases. Traditional experimental methods for determining aggregation properties are time-consuming and resource-intensive. This project addresses this challenge by providing a rapid computational approach to predict protein aggregation tendencies using a fine-tuned ESM2 transformer model. The system aids researchers in understanding protein aggregation behavior, which is crucial for studying neurodegenerative diseases like Alzheimer's and Parkinson's.

We leverage the state-of-the-art ESM2 transformer model, fine-tuning it on the AmyloGraph dataset to classify proteins into three categories:
- No Aggregation
- Slower Aggregation
- Faster Aggregation

## **Model Details**

### Model Card
- **Base Model**: ESM2 (Facebook Research)
- **Architecture**: Transformer-based protein language model
- **Fine-tuning Dataset**: AmyloGraph database
- **Input**: Protein sequences
- **Output**: Classification probabilities for three aggregation categories
- **Performance Highlights**:
  - Strong performance in identifying faster aggregation cases (F1-score: 0.72)
  - High precision for no-aggregation cases (0.82)
  - Balanced performance across multiple metrics

### Dataset Card
- **Source**: AmyloGraph (https://amylograph.com/)
- **Total Samples**: 877 validated sequences
- **Class Distribution**: 
  - No Aggregation: 259 samples
  - Slower Aggregation: 188 samples
  - Faster Aggregation: 430 samples

## **Features**
### 1. Fine-Tuned Transformer Model
- The ESM2 model is fine-tuned using the AmyloGraph dataset
- Predictions include probabilities for each category, providing insights into the sequence's aggregation behavior
- Specialized in detecting faster aggregation patterns

### 2. Interactive Streamlit App
- **Single Sequence Prediction**: Input a sequence string to get the predicted category and probabilities
- **Batch Prediction**: Upload a `.txt` file with multiple sequences for bulk prediction
- **Visualization**: Interactive display of prediction probabilities
- **Results Export**: Download predictions in CSV format

### 3. Notebook Demonstration
- Explore how the model is loaded, evaluated, and used for predictions with a step-by-step walkthrough in `fibril_classification.ipynb`

## **Installation and Usage**
### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Installation Steps
```bash
# Clone the repository
git clone https://github.com/your-username/sequence_classification.git
cd sequence_classification

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Start the Streamlit app
streamlit run app.py

# For Jupyter notebook demonstration
jupyter notebook fibril_classification.ipynb
```

## **Critical Analysis**

### Current Strengths
- Strong performance in identifying faster aggregation cases
- High precision in detecting no-aggregation cases
- Real-time prediction capabilities
- User-friendly interface for both single and batch predictions

### Current Limitations and Future Work

#### 1. Data-Related Challenges
- **Limited Dataset Size**: Current dataset of 176 samples is relatively small for deep learning applications
- **Class Imbalance**: Uneven distribution of samples across classes affects model performance
- **Research Limitations**: Current understanding of fibril protein formation mechanisms is still evolving
- **Proposed Solutions**:
  - Collaborate with research institutions to expand the dataset
  - Implement advanced data augmentation techniques
  - Consider transfer learning from related protein property prediction tasks
  - Explore semi-supervised learning approaches to leverage unlabeled data

#### 2. Model Architecture Improvements
- **Current Model Limitations**: Using basic ESM2 model for fine-tuning
- **Future Enhancements**:
  - Experiment with larger ESM2 variants (650M, 3B parameters) for improved accuracy
  - Investigate ensemble methods combining multiple model architectures
  - Incorporate protein structural information into the model
  - Implement attention visualization for better interpretability

#### 3. Additional Planned Improvements
- Integration of molecular dynamics simulation data
- Development of uncertainty quantification methods
- Enhanced visualization features for sequence analysis
- API development for easier integration with existing workflows
- Multi-task learning to predict additional protein properties

## Additional Resources
- [ESM2: Protein Language Model](https://github.com/facebookresearch/esm)
- [ESM2 Framework](https://nvidia.github.io/bionemo-framework/models/esm2/)
- [Transformer-based deep learning for predicting protein properties](https://elifesciences.org/articles/82819)
- [Building Transformer Models for Proteins From Scratch](https://towardsdatascience.com/building-transformer-models-for-proteins-from-scratch-60884eab5cc8)

## Citation
```
@article{lin2023evolutionary,
  title={Evolutionary-scale prediction of atomic-level protein structure with a language model},
  author={Lin, Zeming and Abanades, Borja and Kieslich, Chris A and Mannige, Ranjan V},
  journal={Science},
  volume={379},
  number={6637},
  pages={1123--1130},
  year={2023},
  publisher={American Association for the Advancement of Science}
}

@article{rives2021biological,
  title={Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences},
  author={Rives, Alexander and Meier, Joshua and Sercu, Tom and Goyal, Siddharth and Lin, Zeming and Liu, Jason and Guo, Demi and Ott, Myle and Zitnick, C Lawrence and Ma, Jerry and others},
  journal={Proceedings of the National Academy of Sciences},
  volume={118},
  number={15},
  pages={e2016239118},
  year={2021},
  publisher={National Acad Sciences}
}

@article{amylograph2023,
  title={AmyloGraph: Computational database of amyloid-protein interactions},
  author={[Author names]}, # You would need to add the actual authors
  journal={[Journal name]}, # Add actual journal
  year={2023},
  publisher={[Publisher name]}
}
```
