# TCN-model
<img width="1686" height="181" alt="image" src="https://github.com/user-attachments/assets/84eac192-d66f-4ded-b361-b72c038bcaca" />
<img width="1919" height="946" alt="image" src="https://github.com/user-attachments/assets/788d8d75-fe28-4a7e-8b5f-58b76b16028d" />

<img width="509" height="463" alt="image" src="https://github.com/user-attachments/assets/9eb5ee32-9bcb-4a43-9742-6ed160792fa7" />


# Hướng dẫn sử dụng Multiple Test Datasets

## 📁 Cấu trúc Files

```
mlops_aws/
├── weather_test_1.py          # Test với weather_test_1.csv
├── weather_test_2.py          # Test với weather_test_2.csv
├── weather_test_3.py          # Test với weather_test_3.csv
├── select_best_model.py       # Tổng hợp kết quả và chọn model tốt nhất
```

## 🎯 Quy trình Test

### Bước 1: Chuẩn bị Test Datasets

Upload 3 file test data lên S3:
- `s3://your-bucket/data/weather_test_1.csv`
- `s3://your-bucket/data/weather_test_2.csv`
- `s3://your-bucket/data/weather_test_3.csv`

### Bước 2: Chạy 3 Tests

Mỗi test sẽ so sánh 2 models trên 1 test dataset và ghi kết quả ra JSON:

```bash
# Test 1
python weather_test_1.py \
  --data-path s3://bucket/data/weather_test_1.csv \
  --model1-path model/weather_model_1.pth \
  --model2-path model/weather_model_2.pth \
  --out-dir model

# Test 2
python weather_test_2.py \
  --data-path s3://bucket/data/weather_test_2.csv \
  --model1-path model/weather_model_1.pth \
  --model2-path model/weather_model_2.pth \
  --out-dir model

# Test 3
python weather_test_3.py \
  --data-path s3://bucket/data/weather_test_3.csv \
  --model1-path model/weather_model_1.pth \
  --model2-path model/weather_model_2.pth \
  --out-dir model
```

**Output sau các bước trên:**
- `model/results_test_1.json`
- `model/results_test_2.json`
- `model/results_test_3.json`

### Bước 3: Chọn Model Tốt Nhất

Chạy script tổng hợp để chọn model có **MSE trung bình thấp nhất** trên cả 3 tests:

```bash
python select_best_model.py \
  --results-dir model \
  --out-dir model
```

**Output:**
- `model/final_model_selection.json` - Kết quả chi tiết
- `model/best_model.pth` - Copy của model tốt nhất

## 📊 Ví dụ Output

### Kết quả từng test (results_test_1.json):
```json
{
  "test_name": "test_1",
  "data_path": "s3://bucket/data/weather_test_1.csv",
  "model1": {
    "path": "model/weather_model_1.pth",
    "mse": 0.024567,
    "latency_sec": 0.1234
  },
  "model2": {
    "path": "model/weather_model_2.pth",
    "mse": 0.019823,
    "latency_sec": 0.1456
  },
  "best_model_path": "model/weather_model_2.pth",
  "best_mse": 0.019823,
  "best_latency": 0.1456
}
```

### Kết quả cuối cùng (final_model_selection.json):
```json
{
  "test_results_summary": {
    "model1": {
      "path": "model/weather_model_1.pth",
      "mse_test_1": 0.024567,
      "mse_test_2": 0.021234,
      "mse_test_3": 0.023456,
      "avg_mse": 0.023086,
      "avg_latency": 0.1234
    },
    "model2": {
      "path": "model/weather_model_2.pth",
      "mse_test_1": 0.019823,
      "mse_test_2": 0.020145,
      "mse_test_3": 0.018967,
      "avg_mse": 0.019645,
      "avg_latency": 0.1456
    }
  },
  "best_model": {
    "winner": "model2",
    "path": "model/weather_model_2.pth",
    "avg_mse": 0.019645,
    "avg_latency": 0.1456
  },
  "selection_criteria": "lowest_avg_mse_across_3_tests_then_latency"
}
```

## 🔧 Tích hợp vào Jenkins/CI/CD

Thêm vào Jenkinsfile:

```groovy
stage('Test on Multiple Datasets') {
    steps {
        script {
            // Test 1
            sh '''
                python weather_test_1.py \
                  --data-path s3://bucket/data/weather_test_1.csv \
                  --model1-path model/weather_model_1.pth \
                  --model2-path model/weather_model_2.pth \
                  --out-dir model
            '''
            
            // Test 2
            sh '''
                python weather_test_2.py \
                  --data-path s3://bucket/data/weather_test_2.csv \
                  --model1-path model/weather_model_1.pth \
                  --model2-path model/weather_model_2.pth \
                  --out-dir model
            '''
            
            // Test 3
            sh '''
                python weather_test_3.py \
                  --data-path s3://bucket/data/weather_test_3.csv \
                  --model1-path model/weather_model_1.pth \
                  --model2-path model/weather_model_2.pth \
                  --out-dir model
            '''
            
            // Select best model
            sh '''
                python select_best_model.py \
                  --results-dir model \
                  --out-dir model
            '''
        }
    }
}
```

## 📈 Tiêu chí chọn Model

1. **MSE trung bình thấp nhất** trên cả 3 test datasets
2. Nếu MSE gần bằng nhau (sai khác < 0.01%), chọn model có **latency thấp hơn**

## ⚠️ Lưu ý

- **Đảm bảo 3 file test data có cùng định dạng** (columns, data types)
- **Model paths phải tồn tại** trước khi chạy tests
- **Output directory phải writable** để ghi JSON files
- Mỗi test file độc lập, có thể chạy song song nếu cần

## 🚀 Quick Start

```bash
# Chạy tất cả trong một lần (sequential)
python weather_test_1.py --data-path data1.csv --model1-path m1.pth --model2-path m2.pth --out-dir model && \
python weather_test_2.py --data-path data2.csv --model1-path m1.pth --model2-path m2.pth --out-dir model && \
python weather_test_3.py --data-path data3.csv --model1-path m1.pth --model2-path m2.pth --out-dir model && \
python select_best_model.py --results-dir model
```

