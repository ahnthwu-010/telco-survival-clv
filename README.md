# Survival-based Customer Lifetime Value Analysis
# Survival-based Customer Lifetime Value Analysis
**Telco Churn · Cox PH · Streamlit**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://telco-survival-clv-dvdhzeflfgllgcrjgtbjdb.streamlit.app/)

> Dự đoán *khi nào* khách hàng rời bỏ và họ đáng giá bao nhiêu — không chỉ *liệu có* rời hay không.

---

## Tại sao dự án này khác biệt

Phần lớn mô hình churn chỉ trả lời câu hỏi nhị phân: rời hay ở lại. Dự án này dùng **Survival Analysis** để trả lời câu hỏi có giá trị kinh doanh thực tế hơn:

- *Khách hàng này có xác suất còn gắn bó sau 12 tháng nữa là bao nhiêu?*
- *Giá trị tài chính còn lại của họ (Remaining CLV) là bao nhiêu, tính từ hôm nay?*
- *Nếu chuyển từ hợp đồng tháng sang hợp đồng năm, CLV thay đổi thế nào?*

---

## Phát hiện chính

**1. Hiệu ứng giá đảo chiều sau ~4 tháng**
Trong 4 tháng đầu, khách hàng trả nhiều tiền hơn có nguy cơ rời bỏ cao hơn (shock giá).
Sau tháng 4, mối quan hệ đảo chiều — khách hàng chi tiêu cao là khách hàng trung thành nhất.
→ *Đừng giảm giá để giữ khách mới. Hãy đầu tư vào onboarding.*

**2. TechSupport không có tác dụng tức thì**
Hazard ratio của TechSupport gần bằng 1.0 ở tháng đầu, nhưng giảm xuống ~0.74 sau 72 tháng.
→ *TechSupport là công cụ giữ khách lâu năm, không phải khách mới.*

**3. Portfolio value: $5.22M — 955 khách hàng at-risk**
955 khách hàng Month-to-month có Remaining CLV cao nhất đại diện cho $236,381
có thể bảo vệ được nếu giữ thêm 10% trong số họ.

---

## Quy trình phân tích

```
1. EDA & Preprocessing
   └── Phát hiện và xử lý TotalCharges dtype issue (11 hàng tenure=0)

2. Kaplan-Meier Survival Curves
   └── Log-rank test: p < 5e-05 (sự khác biệt giữa các nhóm hợp đồng có ý nghĩa thống kê)

3. Cox Proportional Hazards Model
   └── Concordance ban đầu: 0.8156

4. Kiểm định giả định Proportional Hazards (Schoenfeld Residuals)
   └── Phát hiện vi phạm: Contract, MonthlyCharges, TechSupport, OnlineSecurity

5. Stratified Cox PH
   └── Stratify theo Contract để xử lý vi phạm PH của biến này
   └── Concordance: 0.6239 (đúng về mặt thống kê, không phải degradation)

6. Time-Interaction Terms
   └── Thêm MonthlyCharges × log(t) và TechSupport × log(t)
   └── Concordance cuối: 0.8840

7. Remaining CLV Modeling
   └── Conditional survival: S(t | alive at t₀) = S(t) / S(t₀)
   └── Chiết khấu theo thời gian: discount rate 1%/tháng (~12.7%/năm)
   └── CLV = Σ [S(t|t₀) × MonthlyCharges / (1+r)^(t-t₀)]

8. Streamlit App
   └── Portfolio Overview · Customer Lookup · Retention Simulator
```

---

## Kết quả model

| Model | Concordance | Ghi chú |
|---|---|---|
| Cox PH cơ bản | 0.8156 | Vi phạm PH assumption |
| Stratified Cox PH | 0.6239 | Đúng giả định, Contract → strata |
| Stratified + Time-Interaction | **0.8840** | Model cuối, đúng đầy đủ |

---

## Cài đặt và chạy

```bash
git clone https://github.com/YOUR_USERNAME/telco-survival-clv
cd telco-survival-clv
pip install -r requirements.txt
streamlit run app.py
```

---

## Stack

- **Python** · pandas · numpy · matplotlib
- **lifelines** · Kaplan-Meier · Cox PH · Schoenfeld residuals
- **Streamlit** · deployment-ready app
- **Dataset**: [Telco Customer Churn — IBM/Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## Tác giả

Sinh viên Thống kê K49 · Đại học Cần Thơ · 

*Dự án này là phần 1 trong chuỗi portfolio DS, tập trung vào ứng dụng Survival Analysis — một kỹ thuật thống kê ít được triển khai đúng trong cộng đồng DS tự học.*