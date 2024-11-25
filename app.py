import streamlit as st
import torch
import torch.nn.functional as F
import esm

class ProteinClassifier(torch.nn.Module):
    def __init__(self, base_model, num_classes=3):
        super().__init__()
        self.base_model = base_model
        self.dropout = torch.nn.Dropout(0.5)
        self.classification_head = torch.nn.Sequential(
            torch.nn.Linear(base_model.embed_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, num_classes),
        )
        
    def forward(self, tokens):
        outputs = self.base_model(tokens, repr_layers=[6])
        embeddings = outputs["representations"][6][:, 0, :]  # CLS token
        embeddings = self.dropout(embeddings)
        return self.classification_head(embeddings)

@st.cache_resource 
def load_model():
    base_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = ProteinClassifier(base_model)
    model.load_state_dict(torch.load("work_dir/esm_finetuned_model.pt", map_location="cpu"))
    model.eval()
    return model, alphabet

def predict_sequence(model, sequence, batch_converter, device="cpu"):
    data = [("sequence", sequence)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    with torch.no_grad():
        logits = model(tokens)
        probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_label = probabilities.argmax()
    return predicted_label, probabilities

model, alphabet = load_model()
batch_converter = alphabet.get_batch_converter()

label_mapping = {0: "No Aggregation", 1: "Slower Aggregation", 2: "Faster Aggregation"}

st.title("Protein Sequence Classification and Prediction")
st.write("""
This app allows you to classify protein sequences into predefined categories using a fine-tuned deep learning model. 
You can either input a single sequence or upload a file containing multiple sequences for batch prediction. The app will return:
- The predicted label for each sequence.
- The probability distribution across all categories.
Start exploring now to classify your protein sequences efficiently.
""")


option = st.radio(
    "Choose an input method:",
    ("Write a sequence string", "Upload a .txt file")
)

if option == "Write a sequence string":
    sequence = st.text_input("Enter a protein sequence:")
    sequence = sequence.upper()
    if st.button("Predict"):
        if len(sequence) == 0:
            st.warning("Please enter a valid sequence.")
        else:
            predicted_label, probabilities = predict_sequence(model, sequence, batch_converter)
            st.write("### Predicted Results:")
            st.markdown(f"- **Sequence:** {sequence}")
            st.markdown(f"- **Predicted Label:** {label_mapping[predicted_label]}")

            st.write("### Probabilities:")
            st.markdown(f"""
            - **No Aggregation:** {probabilities[0]:.4f}  
            - **Slower Aggregation:** {probabilities[1]:.4f}  
            - **Faster Aggregation:** {probabilities[2]:.4f}  
            """)

elif option == "Upload a .txt file":
    uploaded_file = st.file_uploader("Upload a .txt file containing sequences (one per line):", type="txt")
    if uploaded_file is not None:
        sequences = uploaded_file.read().decode("utf-8").splitlines()
        if st.button("Predict for all sequences"):
            results = []
            for sequence in sequences:
                if len(sequence.strip()) > 0:
                    predicted_label, probabilities = predict_sequence(model, sequence.strip(), batch_converter)
                    results.append((sequence, predicted_label, probabilities))
            
            for seq, label, prob in results:
                st.write("### Predicted Results:")
                st.markdown(f"- **Sequence:** {seq}")
                st.markdown(f"- **Predicted Label:** {label_mapping[label]}")

                st.write("### Probabilities:")
                st.markdown(f"""
                - **No Aggregation:** {prob[0]:.4f}  
                - **Slower Aggregation:** {prob[1]:.4f}  
                - **Faster Aggregation:** {prob[2]:.4f}  
                """)
                st.write("---")
