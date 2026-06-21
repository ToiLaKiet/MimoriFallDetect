import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# 1. Khai báo ma trận nhầm lẫn
# Hàng 0: Non-fall, Hàng 1: Fall
# Cột 0: Dự đoán Non-fall, Cột 1: Dự đoán Fall
cm = np.array([[3440, 17], [32, 90]])

labels = ["Non-fall", "Fall"]

# 2. Tính toán các chỉ số từ ma trận
TN, FP = cm[0, 0], cm[0, 1]
FN, TP = cm[1, 0], cm[1, 1]

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision_fall = TP / (TP + FP)
recall_fall = TP / (TP + FN)
f1_fall = 2 * (precision_fall * recall_fall) / (precision_fall + recall_fall)

print("--- CÁC CHỈ SỐ CƠ BẢN (Cho lớp chính 'Fall') ---")
print(f"Accuracy (Độ chính xác tổng thể): {accuracy:.4f}")
print(f"Precision (Độ chuẩn xác):          {precision_fall:.4f}")
print(f"Recall (Độ nhạy):                 {recall_fall:.4f}")
print(f"F1-Score:                         {f1_fall:.4f}\n")

# 3. Vẽ Heatmap trực quan hóa
plt.figure(figsize=(8, 6))

# Tạo text format hiển thị cả số lượng lẫn tỷ lệ phần trăm trong heatmap
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = [
    "{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)
]
box_labels = [
    f"{v1}\n({v2})" for v1, v2 in zip(group_counts, group_percentages)
]
box_labels = np.asarray(box_labels).reshape(2, 2)

# Vẽ heatmap bằng Seaborn
sns.heatmap(
    cm,
    annot=box_labels,
    fmt="",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    annot_kws={"size": 14},
)

# Cấu hình tiêu đề và nhãn cho trục
plt.title("Confusion Matrix Heatmap (Fall vs Non-fall)", fontsize=16, pad=20)
plt.xlabel("Predicted Label ", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.tight_layout()

# Hiển thị biểu đồ
plt.show()
plt.savefig("confusion_matrix.png")