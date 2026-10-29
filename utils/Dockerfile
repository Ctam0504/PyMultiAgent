FROM python:3.9-alpine
# Cài đặt các công cụ build cơ bản nếu cần (giữ image nhẹ)
RUN apk add --no-cache gcc musl-dev
# Cài sẵn các thư viện hay dùng để tiết kiệm thời gian chạy MAS
RUN pip install --no-cache-dir numpy pandas pytest
WORKDIR /app
# Chặn quyền root để AI không phá hệ thống trong container
RUN adduser -D sandboxuser
USER sandboxuser