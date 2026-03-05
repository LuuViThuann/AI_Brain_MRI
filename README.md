\*ẢNH HỆ THỐNG CHUẨN ĐOÁN U NÃO + 3D TRỰC QUAN

<p align="center">
  <img src="Screen/1.png" width="800">
</p>

<p align="center">
  <img src="Screen/2.png" width="800">
</p>

<p align="center">
  <img src="Screen/3.png" width="800">
</p>

---

- CÁC CÁCH SO SÁCH MÔ HÌNH : <

* Rule-based + thống kê (ngưỡng diện tích u, vị trí, thể tích)

* Grad-CAM / Attention Map (giải thích CNN trực tiếp trên ảnh MRI)

* Decision Tree / Random Forest trên đặc trưng trích xuất

* SHAP / LIME để giải thích đầu ra mô hình

* Prototype-based models (so sánh ca bệnh tương tự)

--

- THÔNG TIN KẾT QUẢ SO SÁNH ==============

2 metric khác nhau:

Dữ liệu 76% là prediction.confidence từ backend (prediction_engine.py)
Dữ liệu 82% là gradcam.attention_score từ XAI (một metric khác hoàn toàn!)

confidence: Mức độ tự tin của CNN về việc có tumor
attention_score: Mức độ tập trung (focus) của CNN vào vùng tumor

---

So sánh:
• CNN thuần: phát hiện nhanh, khó giải thích
• CNN + xAI: phát hiện + minh bạch
• Rule-based: giải thích tốt, độ chính xác thấp hơn
• LLM: hỗ trợ diễn giải, không chẩn đoán

---

- https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

- https://data.mendeley.com/datasets/zwr4ntf94j/4

- https://huggingface.co/datasets/AIOmarRehan/Brain_Tumor_MRI_Dataset?utm_source=chatgpt.com

- https://huggingface.co/datasets/vanhai123/Brain_tumor_detections?utm_source=chatgpt.com

- https://figshare.com/articles/dataset/brain_tumor_dataset/1512427
