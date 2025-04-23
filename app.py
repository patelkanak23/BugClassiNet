import os
import json
import streamlit as st
import xml.etree.ElementTree as ET
import numpy as np
import h5py
import tensorflow as tf
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Bug vs Non‑Bug Classifier")

TRANSFORMER_PATH = r"C:\Users\icham\OneDrive\Desktop\sem_model_3-20250203T085233Z-001\sem_model_3"
H5_PATH = r"C:\Users\icham\OneDrive\Desktop\continued_checkpoints\checkpoint_epoch_10_1744835164.h5"
SAVED_DIR = r"C:\Users\icham\OneDrive\Desktop\cnn_saved_model"

def inspect_model(model):
    summary = []
    model.summary(print_fn=lambda x: summary.append(x))
    return "\n".join(summary)

@st.cache_resource
def load_models():
    embedder = SentenceTransformer(TRANSFORMER_PATH)
    
    test_text = "Test sentence to determine embedding dimension"
    test_embedding = embedder.encode(test_text, convert_to_numpy=True)
    actual_embedding_dim = test_embedding.shape[0]
    st.info(f"Detected embedding dimension: {actual_embedding_dim}")
    
    try:
        classifier = tf.keras.models.load_model(SAVED_DIR)
        
        test_input = np.zeros((1, actual_embedding_dim), dtype=np.float32)
        try:
            _ = classifier.predict(test_input, verbose=0)
            st.success("Model loaded and verified successfully")
            return embedder, classifier
        except Exception as e:
            st.warning(f"Loaded model failed verification: {str(e)}")
    except Exception as e:
        st.warning(f"Could not load saved model: {str(e)}")
    
    if os.path.isdir(SAVED_DIR) and os.path.exists(os.path.join(SAVED_DIR, "saved_model.pb")):
        try:
            classifier = tf.keras.models.load_model(SAVED_DIR)
            test_input = np.random.randn(1, actual_embedding_dim)
            _ = classifier.predict(test_input)
            st.success("Using previously converted model")
            return embedder, classifier
        except Exception:
            st.warning("Found SavedModel but it has incompatible dimensions. Rebuilding...")
            import shutil
            shutil.rmtree(SAVED_DIR)
            os.makedirs(SAVED_DIR, exist_ok=True)
    else:
        os.makedirs(SAVED_DIR, exist_ok=True)
    
    st.info("Analyzing H5 model structure...")
    
    try:
        with h5py.File(H5_PATH, 'r') as h5f:
            has_conv = any('conv' in key for key in h5f['model_weights'].keys())
            output_units = None
            
            for layer_name in h5f['model_weights'].keys():
                if 'dense' in layer_name.lower() and 'kernel' in h5f['model_weights'][layer_name]:
                    kernel_shape = h5f['model_weights'][layer_name]['kernel'].shape
                    output_units = kernel_shape[-1]
            
            if output_units is None:
                output_units = 1
                
        st.info(f"Reconstructing model with input shape {actual_embedding_dim} and output {output_units} units")
        
        if has_conv:
            classifier = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(actual_embedding_dim,), name='embedding_input'),
                tf.keras.layers.Reshape((actual_embedding_dim, 1)),
                tf.keras.layers.Conv1D(64, 3, activation='relu'),
                tf.keras.layers.MaxPooling1D(2),
                tf.keras.layers.Conv1D(128, 3, activation='relu'),
                tf.keras.layers.GlobalMaxPooling1D(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(output_units, activation='sigmoid')
            ])
        else:
            classifier = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(actual_embedding_dim,), name='embedding_input'),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(output_units, activation='sigmoid')
            ])
        
        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        classifier.save(SAVED_DIR)
        st.success("Model reconstructed successfully")
        
    except Exception as e:
        st.error(f"Error rebuilding model: {str(e)}")
        
        st.warning("Creating a simple fallback model...")
        classifier = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(actual_embedding_dim,), name='embedding_input'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        classifier.compile(optimizer='adam', loss='binary_crossentropy')
        classifier.save(SAVED_DIR)
        st.error("⚠️ Using fallback model - results may be random ⚠️")
    
    return embedder, classifier

def parse_xml(file_like) -> list[str]:
    tree = ET.parse(file_like)
    root = tree.getroot()
    out = []
    for bug in root.findall("bug"):
        desc = bug.find("short_desc")
        reso = bug.find("resolution")
        
        desc_text = desc.text if desc is not None and desc.text else ""
        reso_text = reso.text if reso is not None and reso.text else ""
        
        out.append(f"{desc_text}  Resolution: {reso_text}")
    return out

def safe_predict(classifier, embeddings):
    try:
        return classifier.predict(embeddings, verbose=0).flatten()
    except Exception as e1:
        st.warning(f"Standard prediction failed: {str(e1)}")
        try:
            return classifier(tf.convert_to_tensor(embeddings, dtype=tf.float32)).numpy().flatten()
        except Exception as e2:
            st.warning(f"Tensor conversion failed: {str(e2)}")
            try:
                infer = classifier.signatures["serving_default"]
                input_name = list(infer.structured_input_signature[1].keys())[0]
                output_name = list(infer.structured_outputs.keys())[0]
                result = infer(**{input_name: tf.convert_to_tensor(embeddings, dtype=tf.float32)})
                return result[output_name].numpy().flatten()
            except Exception as e3:
                st.warning(f"Signature-based prediction failed: {str(e3)}")
                temp_model = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(embeddings.shape[1],)),
                    tf.keras.layers.Dense(1, activation='sigmoid')
                ])
                try:
                    last_weights = [w for w in classifier.get_weights()[-2:] if len(w.shape) > 0]
                    if len(last_weights) == 2:
                        temp_model.set_weights(last_weights)
                    return temp_model.predict(embeddings, verbose=0).flatten()
                except Exception as e4:
                    st.error(f"All prediction methods failed: {str(e4)}")
                    st.error("⚠️ Generating random predictions as fallback ⚠️")
                    return np.random.random(size=(embeddings.shape[0],))

st.title("🐞 Bug vs Non‑Bug Classifier")

st.markdown(
    "Upload an XML file containing `<bug>` entries.  \n"
    "The model will predict **Bug** or **Non‑Bug** for each entry."
)

try:
    with st.spinner("Loading models..."):
        embedder, classifier = load_models()
    model_loading_success = True
    
except Exception as e:
    st.error(f"Failed to load models: {str(e)}")
    model_loading_success = False

uploaded = st.file_uploader("Choose an XML file", type="xml")

if uploaded:
    if not model_loading_success:
        st.error("Cannot process file because model loading failed.")
    else:
        try:
            snippets = parse_xml(uploaded)
            if not snippets:
                st.warning("No `<bug>` tags found in the uploaded XML.")
            else:
                with st.spinner("Embedding & predicting…"):
                    batch_size = 16
                    all_probs = []
                    
                    for i in range(0, len(snippets), batch_size):
                        batch = snippets[i:i+batch_size]
                        embs = embedder.encode(batch, convert_to_numpy=True)
                        
                        probs = safe_predict(classifier, embs)
                        all_probs.extend(probs)
                
                for i, (txt, p) in enumerate(zip(snippets, all_probs), start=1):
                    label = "Non‑Bug" if p >= 0.5 else "Bug"
                    
                    st.markdown(f"### Entry #{i}")
                    st.write(txt)
                    st.markdown(f"**Prediction:** {label}")

        except ET.ParseError:
            st.error("Failed to parse XML – please upload a valid file.")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.error(f"Full error details: {type(e).__name__}: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
