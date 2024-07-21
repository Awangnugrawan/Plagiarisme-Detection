import os
import re
import pandas as pd
import streamlit as st
import tempfile
import io
import tokenize
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier
import time


def preprocess_text(text):
    # Menghilangkan komentar
    text = re.sub(r"#.*", "", text)

    # Menghilangkan whitespace yang tidak diperlukan
    text = re.sub(r"\s+", " ", text)

    # Menghapus string literal atau menggantinya dengan token
    text = re.sub(r"\".*?\"|\'.*?\'", '"STRING_LITERAL"', text)

    # Menghapus identifier dengan token placeholder
    tokens = tokenize.generate_tokens(io.StringIO(text).readline)
    result = []
    identifiers = {}
    next_id = 0

    for toknum, tokval, _, _, _ in tokens:
        if toknum == tokenize.NAME:
            if tokval not in identifiers:
                identifiers[tokval] = f"var{next_id}"
                next_id += 1
            result.append(identifiers[tokval])
        else:
            result.append(tokval)

    processed_text = " ".join(result)

    return processed_text


# Fungsi untuk menghitung Levenshtein distance
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


# Fungsi untuk mencari file yang terdeteksi plagiarisme menggunakan Levenshtein distance
def find_plagiarized_files_levenshtein(file_paths, threshold):
    plagiarized_files = []
    for i, file_path1 in enumerate(file_paths):
        for file_path2 in file_paths[i + 1 :]:
            with open(file_path1, "r", encoding="utf-8") as f1, open(
                file_path2, "r", encoding="utf-8"
            ) as f2:
                content1 = preprocess_text(f1.read())
                content2 = preprocess_text(f2.read())
                distance = levenshtein_distance(content1, content2)
                max_length = max(len(content1), len(content2))
                if max_length != 0:
                    similarity = 1 - distance / max_length
                    if similarity >= threshold:
                        plagiarized_files.append(
                            (
                                os.path.basename(file_path1),
                                os.path.basename(file_path2),
                                round(similarity * 100, 2),
                                file_path1,
                                file_path2,
                            )
                        )
    return plagiarized_files


class RabinKarp:
    def __init__(self, text, pattern_length, base=256):
        self.text = text
        self.pattern_length = pattern_length
        self.base = base
        self.text_length = len(text)
        self.hash_value = 0

    def calculate_hash(self, start, end):
        k = end - start  # Panjang k-gram
        hash_value = 0
        for i in range(start, end):
            hash_value += ord(self.text[i]) * (self.base ** (k - 1 - (i - start)))
        return hash_value

    def recalculate_hash(self, old_hash, old_char, new_char):
        k = self.pattern_length
        new_hash = self.base * (
            old_hash - ord(old_char) * (self.base ** (k - 1))
        ) + ord(new_char)
        return new_hash

    def get_hash_values(self):
        hash_values = []
        # Pembentukan N-gram dan hitung hash awal untuk N-gram pertama
        self.hash_value = self.calculate_hash(0, self.pattern_length)
        hash_values.append(self.hash_value)

        for i in range(1, self.text_length - self.pattern_length + 1):
            # Hitung hash ulang untuk N-gram berikutnya
            self.hash_value = self.recalculate_hash(
                self.hash_value,
                self.text[i - 1],
                self.text[i + self.pattern_length - 1],
            )
            hash_values.append(self.hash_value)
        return hash_values


def calculate_similarity(text1, text2, k=5):
    rk1 = RabinKarp(text1, k)
    rk2 = RabinKarp(text2, k)

    hash_values1 = rk1.get_hash_values()
    hash_values2 = rk2.get_hash_values()

    common_hashes = len(set(hash_values1).intersection(set(hash_values2)))

    similarity = common_hashes / len(set(hash_values1).union(set(hash_values2)))
    return similarity


# Fungsi untuk mencari file yang terdeteksi plagiarisme menggunakan Rabin-Karp
def find_plagiarized_files_rabinkarp(file_paths, threshold):
    plagiarized_files = []
    for i, file_path1 in enumerate(file_paths):
        for file_path2 in file_paths[i + 1 :]:
            with open(file_path1, "r", encoding="utf-8") as f1, open(
                file_path2, "r", encoding="utf-8"
            ) as f2:
                content1 = preprocess_text(f1.read())
                content2 = preprocess_text(f2.read())

                similarity = calculate_similarity(content1, content2)
                if similarity >= threshold:
                    plagiarized_files.append(
                        (
                            os.path.basename(file_path1),
                            os.path.basename(file_path2),
                            round(similarity * 100, 2),
                            file_path1,
                            file_path2,
                        )
                    )
    return plagiarized_files


# Class estimator untuk Levenshtein
class LevenshteinEstimator(BaseEstimator, ClassifierMixin):
    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        similarities = [self._levenshtein_similarity(x[0], x[1]) for x in X]
        return np.array([[1 - sim, sim] for sim in similarities])

    def _levenshtein_similarity(self, s1, s2):
        distance = levenshtein_distance(s1, s2)
        return 1 - distance / max(len(s1), len(s2))


# Class estimator untuk Rabin-Karp
class RabinKarpEstimator(BaseEstimator, ClassifierMixin):
    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        similarities = [self._rabin_karp_similarity(x[0], x[1]) for x in X]
        return np.array([[1 - sim, sim] for sim in similarities])

    def _rabin_karp_similarity(self, s1, s2):
        similarity = calculate_similarity(s1, s2)
        return similarity


# Fungsi untuk mencari file yang terdeteksi plagiarisme menggunakan Voting Classifier
def find_plagiarized_files_voting(file_paths, threshold):
    plagiarized_files = []
    levenshtein_estimator = LevenshteinEstimator()
    rabin_karp_estimator = RabinKarpEstimator()
    voting_clf = VotingClassifier(
        estimators=[
            ("levenshtein", levenshtein_estimator),
            ("rabin_karp", rabin_karp_estimator),
        ],
        voting="soft",
    )
    X = []
    file_pairs = []
    for i, file_path1 in enumerate(file_paths):
        for file_path2 in file_paths[i + 1 :]:
            with open(file_path1, "r", encoding="utf-8") as f1, open(
                file_path2, "r", encoding="utf-8"
            ) as f2:
                content1 = preprocess_text(f1.read())
                content2 = preprocess_text(f2.read())
                X.append((content1, content2))
                file_pairs.append(
                    (
                        os.path.basename(file_path1),
                        os.path.basename(file_path2),
                        file_path1,
                        file_path2,
                    )
                )
    # Fit the voting classifier with dummy data (required by scikit-learn)
    voting_clf.fit(X, np.zeros(len(X)))
    similarities = voting_clf.predict_proba(X)[:, 1]
    for (file1, file2, path1, path2), similarity in zip(file_pairs, similarities):
        if similarity >= threshold:
            plagiarized_files.append(
                (file1, file2, round(similarity * 100, 2), path1, path2)
            )
    return plagiarized_files


# Fungsi untuk visualisasi hasil plagiarisme
def visualize_results(plagiarized_files, title, language):
    if not plagiarized_files:
        if language == "English":
            st.write("❌ No plagiarism detected.")
        else:
            st.write("❌ Tidak ada plagiarisme yang terdeteksi.")
        return

    df_data = []
    for file1, file2, similarity, path1, path2 in plagiarized_files:
        df_data.append(
            {
                "File 1": file1,
                "File 2": file2,
                "Similarity (%)": similarity,
                "Path 1": path1,
                "Path 2": path2,
            }
        )
    df = pd.DataFrame(df_data)
    df = df.sort_values(by="Similarity (%)", ascending=False)

    for i, row in df.iterrows():
        similarity_formatted = f"**{row['Similarity (%)']}%**"
        if language == "English":
            with st.expander(
                f"📄 {row['File 1']} vs 📄 {row['File 2']} - Similarity: {similarity_formatted}"
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        f"<p style='font-size:24px; text-align:center;'>📄 {row['File 1']}</p>",
                        unsafe_allow_html=True,
                    )
                    with open(row["Path 1"], "r", encoding="utf-8") as f:
                        st.code(f.read(), language="python")
                with col2:
                    st.markdown(
                        f"<p style='font-size:24px; text-align:center;'>📄 {row['File 2']}</p>",
                        unsafe_allow_html=True,
                    )
                    with open(row["Path 2"], "r", encoding="utf-8") as f:
                        st.code(f.read(), language="python")
        else:
            with st.expander(
                f"📄 {row['File 1']} vs 📄 {row['File 2']} - Kemiripan: {similarity_formatted}"
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        f"<p style='font-size:24px; text-align:center;'>📄 {row['File 1']}</p>",
                        unsafe_allow_html=True,
                    )
                    with open(row["Path 1"], "r", encoding="utf-8") as f:
                        st.code(f.read(), language="python")
                with col2:
                    st.markdown(
                        f"<p style='font-size:24px; text-align:center;'>📄 {row['File 2']}</p>",
                        unsafe_allow_html=True,
                    )
                    with open(row["Path 2"], "r", encoding="utf-8") as f:
                        st.code(f.read(), language="python")


uploaded_files_history = []


def main():
    # Menambahkan opsi pemilihan bahasa di sidebar
    language = st.sidebar.selectbox("Language / Bahasa", ["English", "Indonesia"])

    if language == "English":
        st.header("🏠 Welcome to Plagiarism Detection for Python Code!")
        st.markdown("This web application helps you detect plagiarism in Python code.")
    else:
        st.header("🏠 Selamat Datang di Deteksi Plagiarisme untuk Kode Python!")
        st.markdown(
            "Aplikasi web ini membantu Anda mendeteksi plagiarisme dalam kode Python."
        )

    with st.expander(
        "🔔 Plagiarism Detection Steps ⬇️"
        if language == "English"
        else "🔔 Langkah-langkah Deteksi Plagiarisme ⬇️"
    ):
        if language == "English":
            st.subheader("📂 Upload Your Python Files")
            st.markdown(
                "Upload two or more Python files that you want to compare for plagiarism. Click the **Browse files** button on the side ⬅️ and select the Python files from your computer."
            )

            st.subheader("🔢 Set Similarity Threshold")
            st.markdown(
                "Choose a similarity threshold percentage using the slider. Files with similarity percentages equal to or greater than this threshold will be considered plagiarized."
            )

            st.subheader("🚀 Detect Plagiarism")
            st.markdown(
                "Once you have uploaded your files and set the similarity threshold, click the **Detect Plagiarism** button to start the detection process."
            )

            st.subheader("📝 Plagiarism Detection Results")
            st.markdown(
                "After the detection process is complete, the results will be displayed in different tabs. Each tab represents a different method of plagiarism detection."
            )
        else:
            st.subheader("📂 Unggah File Python")
            st.markdown(
                "Unggah dua atau lebih file Python yang ingin  dibandingkan untuk plagiarisme. Klik tombol **Telusuri file** di samping ⬅️ dan pilih file Python dari local komputer ."
            )

            st.subheader("🔢 Atur Ambang Batas Kemiripan")
            st.markdown(
                "Pilih persentase ambang batas kemiripan menggunakan slider. File dengan persentase kemiripan sama dengan atau lebih besar dari ambang batas ini akan dianggap plagiarisme."
            )

            st.subheader("🚀 Deteksi Plagiarisme")
            st.markdown(
                "Setelah mengunggah file dan mengatur ambang batas kemiripan, klik tombol **Deteksi Plagiarisme** untuk memulai proses deteksi."
            )

            st.subheader("📝 Hasil Deteksi Plagiarisme")
            st.markdown(
                "Setelah proses deteksi selesai, hasilnya akan ditampilkan di berbagai tab. Setiap tab mewakili metode deteksi plagiarisme yang berbeda."
            )

    st.sidebar.title(
        "🔍 Plagiarism Detection Settings"
        if language == "English"
        else "🔍 Pengaturan Deteksi Plagiarisme"
    )

    # Menyimpan riwayat file yang diupload
    if uploaded_files_history:
        st.sidebar.write(
            "Uploaded Files History:"
            if language == "English"
            else "Riwayat File yang Diupload:"
        )
        for file_name in uploaded_files_history:
            st.sidebar.write(file_name)

    uploaded_files = st.sidebar.file_uploader(
        "📂 Upload Python Files" if language == "English" else "📂 Unggah File Python",
        type="py",
        accept_multiple_files=True,
    )

    # Menambahkan file yang baru diupload ke dalam riwayat
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in uploaded_files_history:
                uploaded_files_history.append(uploaded_file.name)

    threshold = (
        st.sidebar.selectbox(
            (
                "🔢 Similarity Threshold (%)"
                if language == "English"
                else "🔢 Ambang Batas Kemiripan (%)"
            ),
            range(0, 101, 5),
        )
        / 100
    )

    # Fungsi deteksi plagiarisme
    if st.sidebar.button(
        "🚀 Detect Plagiarism" if language == "English" else "🚀 Deteksi Plagiarisme"
    ):
        if not uploaded_files:
            st.sidebar.error(
                "❗ Please upload at least two files."
                if language == "English"
                else "❗ Silakan unggah setidaknya dua file."
            )
        elif len(uploaded_files) < 2:
            st.sidebar.error(
                "❗ Please upload at least two files."
                if language == "English"
                else "❗ Silakan unggah setidaknya dua file."
            )
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_paths = []
                for uploaded_file in uploaded_files:
                    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_file_path, "wb") as temp_file:
                        temp_file.write(uploaded_file.getbuffer())
                    file_paths.append(temp_file_path)

                progress_placeholder = st.empty()
                progress_bar = progress_placeholder.progress(0)
                progress_text = st.empty()

                total_steps = 3  # Number of detection functions
                durations = []

                for step in range(1, total_steps + 1):
                    start_time = time.time()
                    if step == 1:
                        results_rabinkarp = find_plagiarized_files_rabinkarp(
                            file_paths, threshold
                        )
                    elif step == 2:
                        results_levenshtein = find_plagiarized_files_levenshtein(
                            file_paths, threshold
                        )
                    elif step == 3:
                        results_voting = find_plagiarized_files_voting(
                            file_paths, threshold
                        )
                    end_time = time.time()
                    duration = end_time - start_time
                    durations.append(duration)

                    progress = int(step / total_steps * 100)
                    progress_bar.progress(progress)
                    progress_text.write(
                        f"Loading... {progress}%"
                        if language == "English"
                        else f"Memuat... {progress}%"
                    )
                    time.sleep(1)  # Simulate some processing time

                st.success(
                    "✅ Plagiarism detection completed."
                    if language == "English"
                    else "✅ Deteksi plagiarisme selesai."
                )
                progress_placeholder.empty()  # Remove the progress bar
                progress_text.empty()  # Remove the progress text

                st.header(
                    "📝 Plagiarism Detection Results"
                    if language == "English"
                    else "📝 Hasil Deteksi Plagiarisme"
                )
                st.subheader(
                    f"**Number of Python files compared: {len(file_paths)}**"
                    if language == "English"
                    else f"**Jumlah file Python yang dibandingkan: {len(file_paths)}**"
                )

                def format_duration(duration):
                    if duration > 60:
                        minutes = int(duration // 60)
                        seconds = duration % 60
                        return (
                            f"{minutes} minutes {seconds:.2f} seconds"
                            if language == "English"
                            else f"{minutes} menit {seconds:.2f} detik"
                        )
                    else:
                        return (
                            f"{duration:.2f} seconds"
                            if language == "English"
                            else f"{duration:.2f} detik"
                        )

                st.write(
                    f"**Rabin-Karp Duration:** {format_duration(durations[0])}, detected {len(results_rabinkarp)} plagiarized file."
                    if language == "English"
                    else f"**Durasi Rabin-Karp:** {format_duration(durations[0])}, terdeteksi {len(results_rabinkarp)} file yang plagiat."
                )
                st.write(
                    f"**Levenshtein Distance Duration:** {format_duration(durations[1])}, detected {len(results_levenshtein)} plagiarized file."
                    if language == "English"
                    else f"**Durasi Jarak Levenshtein:** {format_duration(durations[1])}, terdeteksi {len(results_levenshtein)} file yang plagiat."
                )
                st.write(
                    f"**Voting Classifier Duration:** {format_duration(durations[2])}, detected {len(results_voting)} plagiarized file."
                    if language == "English"
                    else f"**Durasi Voting Classifier:** {format_duration(durations[2])}, terdeteksi {len(results_voting)} file yang plagiat."
                )
                tab1, tab2, tab3 = st.tabs(
                    [
                        (
                            "🔍 Levenshtein Distance"
                            if language == "English"
                            else "🔍 Jarak Levenshtein"
                        ),
                        "🔍 Rabin-Karp",
                        "🔍 Voting Classifier",
                    ]
                )

                with tab1:
                    st.subheader(
                        "Levenshtein Distance Results"
                        if language == "English"
                        else "Hasil Levenshtein Distance"
                    )
                    visualize_results(
                        results_levenshtein,
                        (
                            "Levenshtein Distance Plagiarism Detection Results"
                            if language == "English"
                            else "Hasil Deteksi Plagiarisme Jarak Levenshtein"
                        ),
                        language,
                    )

                with tab2:
                    st.subheader(
                        "Rabin-Karp Results"
                        if language == "English"
                        else "Hasil Rabin-Karp"
                    )
                    visualize_results(
                        results_rabinkarp,
                        (
                            "Rabin-Karp Plagiarism Detection Results"
                            if language == "English"
                            else "Hasil Deteksi Plagiarisme Rabin-Karp"
                        ),
                        language,
                    )

                with tab3:
                    st.subheader(
                        "Voting Classifier Results"
                        if language == "English"
                        else "Hasil Voting Classifier"
                    )
                    visualize_results(
                        results_voting,
                        (
                            "Voting Classifier Plagiarism Detection Results"
                            if language == "English"
                            else "Hasil Deteksi Plagiarisme Voting Classifier"
                        ),
                        language,
                    )

    st.sidebar.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "[© Awang Mulya Nugrawan](https://www.linkedin.com/in/awang-nugrawan/)"
    )


if __name__ == "__main__":
    st.set_page_config(
        page_title="Plagiarism Detection",
        page_icon="https://cdn-icons-png.flaticon.com/128/2621/2621303.png",
        layout="wide",
    )
    main()
