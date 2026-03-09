import os
import io
import zipfile
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image
import pdfplumber
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10

# -----------------------------
# Streamlit App Title
# -----------------------------
st.set_page_config(page_title="ML Algorithm Identifier", layout="wide")
st.title("🤖 Identification of Algorithm from Dataset using AIML")

# -----------------------------
# File Uploader + Demo Mode
# -----------------------------
uploaded_files = st.file_uploader(
    "📂 Upload one or more datasets (.csv, .jpg, .jpeg, .pdf, .zip)",
    type=["csv", "jpg", "png", "jpeg", "pdf", "zip"],
    accept_multiple_files=True
)

demo_choice = st.selectbox(
    "Or select a Demo Dataset:",
    [
        "None",
        "MNIST (10k sample)",
        "Fashion-MNIST (10k sample)",
        "CIFAR-10 (10k sample)",
        "Cats vs Dogs (Demo Image)",
        "PDF Text Classification (Demo)"
    ]
)

df, X, y, file_type, data_kind, preview_images = None, None, None, None, None, []

# -----------------------------
# Load Demo Dataset
# -----------------------------
if demo_choice != "None":
    if demo_choice == "MNIST (10k sample)":
        (X, y), _ = mnist.load_data()
        X = X[:10000].reshape(10000, -1)
        y = y[:10000]
        df = pd.DataFrame(X)
        df["target"] = y
        file_type, data_kind = "demo", "image"
        st.success("✅ MNIST demo dataset loaded!")

        sample_indices = np.random.choice(len(y), size=5, replace=False)
        for idx in sample_indices:
            img_array = X[idx].reshape(28, 28)
            preview_images.append((img_array, f"Sample {idx}", str(y[idx])))

    elif demo_choice == "Fashion-MNIST (10k sample)":
        (X, y), _ = fashion_mnist.load_data()
        X = X[:10000].reshape(10000, -1)
        y = y[:10000]
        df = pd.DataFrame(X)
        df["target"] = y
        file_type, data_kind = "demo", "image"
        st.success("✅ Fashion-MNIST demo dataset loaded!")

        sample_indices = np.random.choice(len(y), size=5, replace=False)
        for idx in sample_indices:
            img_array = X[idx].reshape(28, 28)
            preview_images.append((img_array, f"Sample {idx}", str(y[idx])))

    elif demo_choice == "CIFAR-10 (10k sample)":
        (X, y), _ = cifar10.load_data()
        X = X[:10000].reshape(10000, -1)
        y = y[:10000].flatten()
        df = pd.DataFrame(X)
        df["target"] = y
        file_type, data_kind = "demo", "image"
        st.success("✅ CIFAR-10 demo dataset loaded!")

        sample_indices = np.random.choice(len(y), size=5, replace=False)
        for idx in sample_indices:
            img_array = X[idx].reshape(32, 32, 3)
            preview_images.append((img_array, f"Sample {idx}", str(y[idx])))

    # -----------------------------
    # Custom Demo: Cats vs Dogs
    # -----------------------------
    elif demo_choice == "Cats vs Dogs (Demo Image)":
        num_samples = 1000
        X = np.random.randint(0, 256, (num_samples, 28*28))
        y = np.array(["cat"] * (num_samples // 2) + ["dog"] * (num_samples // 2))

        df = pd.DataFrame(X)
        df["target"] = y
        file_type, data_kind = "demo", "image"
        st.success("✅ Cats vs Dogs demo dataset loaded! (1000 samples)")

        for i in range(5):
            img_array = X[i].reshape(28, 28)
            preview_images.append((img_array, f"Demo_{i}", y[i]))

    # -----------------------------
    # Custom Demo: PDF Text Classification
    # -----------------------------
    elif demo_choice == "PDF Text Classification (Demo)":
        texts = []
        labels = []
        categories = ["Agriculture", "Sports", "Technology"]

        # generate 50 sample docs
        for i in range(50):
            if i % 3 == 0:
                texts.append(f"Agriculture report about farming methods and crop yield improvements {i}")
                labels.append("Agriculture")
            elif i % 3 == 1:
                texts.append(f"Sports news article covering football, cricket and Olympic updates {i}")
                labels.append("Sports")
            else:
                texts.append(f"Technology article on artificial intelligence and software trends {i}")
                labels.append("Technology")

        df = pd.DataFrame({"text": texts, "category": labels})
        file_type, data_kind = "demo", "text"
        st.success("✅ PDF text classification demo dataset loaded! (50 documents)")

        st.subheader("📄 Sample Demo Documents (5 shown)")
        for i, row in df.head(5).iterrows():
            st.text_area(f"Doc {i+1}", row["text"], height=100)

# -----------------------------
# Load Uploaded Dataset
# -----------------------------
elif uploaded_files:
    file = uploaded_files[0]
    file_type = file.name.split(".")[-1].lower()

    try:
        if file_type == "csv":
            df = pd.read_csv(file)
            data_kind = "csv"

        elif file_type in ["jpg", "png", "jpeg"]:
            img = Image.open(file).convert("L").resize((28, 28))
            data = np.array(img).reshape(1, -1)
            df = pd.DataFrame(data)
            df["target"] = [0]
            data_kind = "image"
            preview_images.append((img, file.name, "label: 0"))

        elif file_type == "zip":
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(file, "r") as z:
                    z.extractall(tmpdir)
                imgs, labels = [], []
                all_files = []
                for root, _, files in os.walk(tmpdir):
                    for f in files:
                        if f.lower().endswith((".jpg", ".png", ".jpeg")):
                            all_files.append(os.path.join(root, f))

                # Auto-sample if too many images
                max_samples = 2000
                if len(all_files) > max_samples:
                    st.warning(f"⚠️ Large dataset detected ({len(all_files)} images). Using {max_samples} random samples.")
                    all_files = list(np.random.choice(all_files, size=max_samples, replace=False))

                for fpath in all_files:
                    img = Image.open(fpath).convert("L").resize((28, 28))
                    imgs.append(np.array(img).flatten())

                    # Label from folder name if available
                    label = os.path.basename(os.path.dirname(fpath))
                    if not label:
                        label = "unknown"
                    labels.append(label)

                    if len(preview_images) < 5:
                        preview_images.append((img, os.path.basename(fpath), label))

                if imgs:
                    df = pd.DataFrame(imgs)
                    df["target"] = labels
                    data_kind = "image"

        elif file_type == "pdf":
            texts = []
            targets = []
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    texts.append(text)

                    # Extract label from page using "Category:" pattern (case-insensitive)
                    label = None
                    for line in text.split("\n"):
                        if line.strip().lower().startswith("category:"):
                            # take everything after first colon
                            label = line.split(":", 1)[1].strip()
                            break
                    if not label:
                        label = "Unknown"
                    targets.append(label)

            df = pd.DataFrame({"text": texts, "target": targets})
            data_kind = "text"

        if df is not None:
            st.success("✅ Dataset successfully loaded!")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")

# -----------------------------
# Show Meta Features
# -----------------------------
if df is not None:
    st.subheader("📊 Extracted Meta Features")
    try:
        meta = {
            "type": "Tabular Data" if data_kind != "text" else "Text Data",
            "num_rows": df.shape[0],
            "num_columns": df.shape[1],
            "num_numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
            "num_categorical_columns": len(df.select_dtypes(include=["object"]).columns),
            "missing_values_total": int(df.isnull().sum().sum())
        }
        st.json(meta)
    except Exception as e:
        st.error(f"Error extracting metadata: {e}")

    # -----------------------------
    # CSV Preview
    # -----------------------------
    if data_kind == "csv":
        st.subheader("📑 Sample Rows from CSV (first 5)")
        st.dataframe(df.head(5))

    # -----------------------------
    # Image Preview
    # -----------------------------
    if data_kind == "image" and preview_images:
        st.subheader("🖼️ Sample Images (5 max)")
        n_show = min(5, len(preview_images))
        cols = st.columns(n_show)
        for i, (img, fname, lbl) in enumerate(preview_images[:n_show]):
            with cols[i]:
                st.image(img, caption=f"{fname} | {lbl}", use_container_width=True)

    # -----------------------------
    # Text Preview (PDF or Demo PDF)
    # -----------------------------
    if data_kind == "text":
        st.subheader("📄 Sample Text Data (first 5 docs)")
        for i, row in df.head(5).iterrows():
            st.text_area(f"Doc {i+1}", row.get("text", ""), height=120)

    # -----------------------------
    # Automatic Target Column Selection
    # -----------------------------
    # Automatically pick the last column as the target column for all datasets
    target_col = df.columns[-1]
    st.success(f"🎯 Automatically selected target column: {target_col}")

    # -----------------------------
    # Training & Evaluation
    # -----------------------------
    if target_col:
        st.subheader("⚙️ Training & Evaluating Algorithms")

        y = df[target_col]

        # --- Convert features to numeric ---
        if data_kind == "text":
            vectorizer = TfidfVectorizer(max_features=500)
            X = vectorizer.fit_transform(df["text"].fillna("")).toarray()
        else:
            X = df.drop(columns=[target_col])
            if X.select_dtypes(include=['object']).shape[1] > 0:
                X = X.apply(LabelEncoder().fit_transform)

        # --- Ensure target labels are numeric ---
        if y.dtype == 'object' or str(y.dtype).startswith('category'):
            y = LabelEncoder().fit_transform(y)

        # handle small datasets gracefully
        if len(y) < 2 or len(np.unique(y)) < 2:
            st.error("❌ Insufficient classes/samples to train models.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            algorithms = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(),
                "SVM": SVC(),
                "KNN": KNeighborsClassifier(),
                "Naive Bayes": GaussianNB(),
                "Decision Tree": DecisionTreeClassifier(random_state=42)
            }

            results = {}
            for name, model in algorithms.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    results[name] = acc
                except Exception as e:
                    st.warning(f"{name} failed: {e}")

            if results:
                best_model = max(results, key=results.get)
                st.success(f"✅ Recommended: {best_model} (Accuracy: {results[best_model]:.2f})")

                results_df = pd.DataFrame({
                    "Algorithm": list(results.keys()),
                    "Accuracy": list(results.values())
                })
                fig = px.bar(
                    results_df,
                    x="Algorithm",
                    y="Accuracy",
                    color="Algorithm",
                    text="Accuracy",
                    title="📈 Algorithm Accuracy Comparison"
                )
                fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig.update_layout(yaxis=dict(range=[0, 1]))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("❌ No models were trained successfully.")
