#MetalBind_PATN: Enhancing Metal-Binding Residue Classification Using Protein Language Model Embeddings and Positional Attention Transformer Network
This paper introduces the MetalBind_PATN model, which achieves substantial improvements in predicting metal-ion-binding sites. By using a transformer-based neural network architecture and advanced features such as multi-scale residue information and ProtTrans embeddings, the model significantly enhances the prediction accuracy compared to previous methods. The results demonstrate impressive evaluation metrics, such as an AUC of 88.28% and an accuracy of 90.45%. The model’s performance shows considerable improvements over baseline methods, particularly in recall (46%) and MCC (12%). These advancements make MetalBind_PATN a valuable tool for metal-binding site prediction in structural bioinformatics.
## Fig. 1: Our Comprehensive Research Workflow:

![Figure 1](Figure_Archtectural_1.png)

# Methodology 


## Quick Start

### Step 1: Generate Data Features
Navigate to the 'data' folder and use the FASTA file to generate additional data features that are saved in the 'dataset' folder..

**Example usage:**

```bash
python get_ProtTrans.py -in "Your FASTA file folder" -out "The destination folder of your output"
python get_tape.py -in "Your FASTA file folder" -out "The destination folder of your output"
python get_esm.py "Pretrained model of ESM" "Your FASTA file folder" "The destination folder of your output" --repr_layers 33 --include per_tok
python get_ProstT5.py -in "Your FASTA file folder" -out "The destination folder of your output"
```
### Step 2: Generate Dataset Using Data Features
1. **Run `length_change.ipynb`:**
   - Open the `length_change.ipynb` file and specify the following:
     - The proper paths for the training and testing datasets.
     - The feature type: use `'pt'` for ProtTrans, `'esm'` for ESM, and `'tape'` for TAPE.
     - Set the desired sequence length for the study.

2. **Run `Concatenate.ipynb`:**
   - Execute the `Concatenate.ipynb` file to concatenate all protein sequences. This step will produce the following output files:
     - `train_data.npy`: Contains the training data.
     - `train_labels.npy`: Contains the corresponding training labels.
     - `testing_data.npy`: Contains the testing data.
     - `testing_labels.npy`: Contains the corresponding testing labels.
    
### Step 3: Execute Prediction
1. **Navigate to the code folder:**
   - Change your directory to the `code` folder where the prediction model is located.

2. **Run the Model:**
   - Open the `mCNN_Sodium.ipynb` file in Jupyter Notebook.
   - Execute the cells in the notebook to run the model and make predictions based on your dataset.

## References
1.	UniProt: the Universal Protein knowledgebase in 2023. Nucleic Acids Research, 2023. 51(D1): p. D523-D531.
2.	Elnaggar, A., et al., ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning. IEEE Trans Pattern Anal Mach Intell, 2022. 44(10): p. 7112-7127.
