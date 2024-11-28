Trong file Dockerfile, có hai biến môi trường liên quan đến Kaggle
```
ENV KAGGLE_USERNAME=<YOUR_KAGGLE_USERNAME>
ENV KAGGLE_KEY=<YOUR_KAGGLE_API_KEY>
```
Chạy lệnh sau trong thư mục chính của repository để xây dựng Docker image:
```
docker build -t chunking_service .
```
Sau khi Docker image đã được xây dựng, chạy container bằng lệnh
```
docker run -d -p 8000:8000 --name chunking_service chunking_service
```
