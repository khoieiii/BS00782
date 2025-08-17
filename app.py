import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Phân tích và Dự đoán Doanh số Bán hàng", layout="wide")

st.title("Phân tích và Dự đoán Doanh số Bán hàng")
st.markdown("Một ứng dụng Streamlit để phân tích dữ liệu bán hàng và dự đoán doanh thu.")
st.header("1. Tải và Chuẩn bị Dữ liệu")
st.markdown("Đọc file `sales_data.csv` và hiển thị 5 dòng đầu tiên.")

try:
    df = pd.read_csv('sales_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    st.dataframe(df.head())
except FileNotFoundError:
    st.error("Lỗi: Không tìm thấy file `sales_data.csv`. Vui lòng đảm bảo file này nằm trong cùng thư mục với `app.py`.")
    st.stop()
st.header("2. Phân tích Dữ liệu với Biểu đồ")
st.markdown("Sử dụng Matplotlib và Seaborn để trực quan hóa dữ liệu.")

sns.set_style('whitegrid')

# Biểu đồ 1: Doanh thu theo Category
st.subheader("Doanh thu theo Danh mục Sản phẩm")
revenue_by_category = df.groupby('Category')['Revenue'].sum().reset_index()
fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.barplot(x='Category', y='Revenue', data=revenue_by_category, ax=ax1, palette='viridis')
ax1.set_xlabel('Category')
ax1.set_ylabel('Total Revenue')
ax1.set_title('Total Revenue by Product Category')
plt.xticks(rotation=45)
st.pyplot(fig1)
st.subheader("Số lượng Sản phẩm đã bán theo Địa điểm")
units_by_location = df.groupby('Location')['UnitsSold'].sum().reset_index()
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.barplot(x='Location', y='UnitsSold', data=units_by_location, ax=ax2, palette='plasma')
ax2.set_xlabel('Location')
ax2.set_ylabel('Total Units Sold')
ax2.set_title('Total Units Sold by Location')
plt.xticks(rotation=45)
st.pyplot(fig2)
st.subheader("Doanh thu theo Tháng trong năm 2023")
revenue_by_month = df.groupby('Month')['Revenue'].sum().reset_index()
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.lineplot(x='Month', y='Revenue', data=revenue_by_month, marker='o', ax=ax3)
ax3.set_xlabel('Month')
ax3.set_ylabel('Total Revenue')
ax3.set_title('Total Revenue by Month in 2023')
ax3.set_xticks(range(1, 13))
ax3.grid(True)
st.pyplot(fig3)
st.subheader("Top 5 Sản phẩm có Doanh thu cao nhất")
revenue_by_product = df.groupby('ProductID')['Revenue'].sum().nlargest(5).reset_index()
fig4, ax4 = plt.subplots(figsize=(8, 6))
sns.barplot(x='ProductID', y='Revenue', data=revenue_by_product, ax=ax4, palette='rocket')
ax4.set_xlabel('Product ID')
ax4.set_ylabel('Total Revenue')
ax4.set_title('Top 5 Products by Revenue')
plt.xticks(rotation=45)
st.pyplot(fig4)

st.subheader("Phân phối Giá sản phẩm (Unit Price)")
fig5, ax5 = plt.subplots(figsize=(8, 6))
sns.histplot(df['UnitPrice'], bins=10, kde=True, ax=ax5)
ax5.set_xlabel('Unit Price')
ax5.set_ylabel('Frequency')
ax5.set_title('Distribution of Unit Price')
st.pyplot(fig5)

st.header("3. Dự đoán Doanh thu với Hồi quy Tuyến tính")
st.markdown("Chúng ta sẽ xây dựng một mô hình hồi quy tuyến tính để dự đoán doanh thu.")

X = df[['Category', 'Location', 'UnitsSold', 'UnitPrice', 'Month']]
y = df['Revenue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
categorical_features = ['Category', 'Location']
numerical_features = ['UnitsSold', 'UnitPrice', 'Month']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
st.write(f'**Mean Squared Error (MSE):** {mse:.2f}')
st.write(f'**Root Mean Squared Error (RMSE):** {rmse:.2f}')
st.write(f'**R² Score:** {r2:.2f}')

st.subheader("Doanh thu Thực tế vs. Dự đoán")
fig_pred, ax_pred = plt.subplots(figsize=(8, 6))
ax_pred.scatter(y_test, y_pred, color='#1f77b4', edgecolor='black', alpha=0.6)
ax_pred.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax_pred.set_xlabel('Actual Revenue')
ax_pred.set_ylabel('Predicted Revenue')
ax_pred.set_title('Actual vs Predicted Revenue (Linear Regression)')
st.pyplot(fig_pred)

st.subheader("Biểu đồ Phần dư (Residual Plot)")
residuals = y_test - y_pred
fig_res, ax_res = plt.subplots(figsize=(8, 6))
ax_res.scatter(y_pred, residuals, color='#d62728', edgecolor='black', alpha=0.6)
ax_res.axhline(y=0, color='r', linestyle='--', lw=2)
ax_res.set_xlabel('Predicted Revenue')
ax_res.set_ylabel('Residuals')
ax_res.set_title('Residual Plot')
st.pyplot(fig_res)

st.subheader("Diễn giải Hệ số của Mô hình")
try:
    ohe = model.named_steps['preprocessor'].named_transformers_['cat']
    cat_features = ohe.get_feature_names_out(['Category', 'Location'])
    num_features = ['UnitsSold', 'UnitPrice', 'Month']
    all_features = np.concatenate([num_features, cat_features])
    coefficients = model.named_steps['regressor'].coef_
    intercept = model.named_steps['regressor'].intercept_

    coef_df = pd.DataFrame({'Feature': all_features, 'Coefficient': coefficients})
    st.write("Hệ số của mô hình:")
    st.dataframe(coef_df)
    st.write(f"Hệ số chặn (Intercept): **{intercept:.2f}**")
except Exception as e:
    st.error(f"Lỗi khi diễn giải hệ số: {e}")

st.markdown("---")
st.info("Ứng dụng được xây dựng với Streamlit và triển khai qua GitHub.")
