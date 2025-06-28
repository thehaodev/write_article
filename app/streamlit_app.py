import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import streamlit as st

from author_profiling.preprocess import stem_lema
from author_profiling.count_word import count_word
from author_profiling.similarity_utils import cosine_similarity_sklearn



THRESHOLD = 0.8  # Ngưỡng phân biệt cùng tác giả

def compare_authors():
    """
    Hàm chính để so sánh tác giả của hai văn bản.
    - Đọc file alice.txt
    - Đọc file văn bản cần kiểm tra
    - Tiền xử lý và tính n-gram
    - Tính Cosine Similarity
    - In kết quả so sánh
    """
    # Đọc file alice.txt
    with open("data/alice.txt", "r", encoding="utf-8") as f:
        alice_text = f.read()

    # Đọc file văn bản cần kiểm tra
    with open("test_text.txt", "r", encoding="utf-8") as f:
        test_text = f.read()

    # Tiền xử lý và tính n-gram
    alice_tokens = stem_lema(alice_text)
    test_tokens = stem_lema(test_text)

    alice_profile = count_word(alice_tokens, n=2)
    test_profile = count_word(test_tokens, n=2)

    # Tính Cosine Similarity
    similarity = cosine_similarity_sklearn(alice_profile, test_profile)

    # In kết quả
    print(f"Cosine Similarity giữa test_text và alice.txt: {similarity:.4f}")

    return THRESHOLD


def run_streamlit():
    # Giao diện chính
    st.set_page_config(page_title="Author Profiling - Lewis Carroll", layout="centered")

    st.title("📄 Author Profiling - Lewis Carroll")

    st.markdown("""
    Ứng dụng kiểm tra xem văn bản bạn cung cấp có khả năng cùng tác giả với **Lewis Carroll** hay không, 
    dựa trên kỹ thuật phân tích văn bản và so sánh đặc trưng n-gram.
    """)

    st.header("🔍 Bước 1: Chọn file văn bản cần kiểm tra")
    uploaded_file = st.file_uploader("Chọn file (.txt)", type="txt")

    if uploaded_file is not None:
        text = uploaded_file.read().decode('utf-8')
        st.text_area("📑 Nội dung văn bản:", text, height=200)

        st.header("⚙️ Bước 2: Kết quả kiểm tra")

        # Đây là chỗ bạn sẽ gọi các hàm xử lý logic n-gram, similarity...
        # Mình chỉ để placeholder cho bạn chèn logic sau:
        
        if st.button("Kiểm tra"):
            # TODO: Gọi hàm xử lý n-gram, tính cosine similarity
            # Ví dụ:
            # similarity = your_function_to_compare(text)

            similarity = 0.85  # Giá trị mẫu, bạn thay bằng giá trị thực

            st.write(f"**Cosine Similarity:** {similarity:.4f}")

            if similarity > 0.8:
                st.success("✅ Kết luận: Có khả năng cùng tác giả Lewis Carroll.")
            else:
                st.warning("⚠️ Kết luận: Khác tác giả Lewis Carroll.")


run_streamlit()