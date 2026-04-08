FROM python:3.13-slim

WORKDIR /app

# 依存パッケージをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt streamlit plotly

# アプリコードをコピー
COPY config/ config/
COPY src/ src/
COPY app_pages/ app_pages/
COPY app.py .
COPY run_scraper.py .
COPY run_train_v3.py .
COPY run_predict_sunday.py .
COPY run_ev_backtest.py .

# データ・モデル用のディレクトリ
RUN mkdir -p data/raw data/processed models

# Streamlitの設定
RUN mkdir -p /root/.streamlit
COPY .streamlit/config.toml /root/.streamlit/config.toml

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
