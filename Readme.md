# 🤖 AI Crypto Trading Bot

This project is an end-to-end automated trading bot for cryptocurrencies (specifically ETH/USDT). It uses a deep learning model (LSTM) to predict short-term price movements and provides trading signals via a real-time API. The entire MLOps pipeline is managed, from data collection and model training on AWS SageMaker to deployment and live inference.

## Table of Contents
- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Workflow](#workflow)
- [API Endpoints](#api-endpoints)
- [Disclaimer](#disclaimer)

## Project Overview

The core of this project is a predictive model that analyzes 1-minute candlestick data (OHLCV) for ETH/USDT. It learns patterns from historical data to generate a "BUY" or "HOLD" signal based on the likelihood of a price increase in the near future.

The project is designed with a clear separation of concerns:
1.  **Data Engineering**: Scripts to fetch and prepare large datasets.
2.  **ML Training**: A robust training and deployment pipeline using AWS SageMaker.
3.  **Inference**: A real-time API built with FastAPI to serve live predictions and stream data.

## Architecture

The project follows a standard MLOps workflow:

1.  **📥 Data Ingestion**: `history.py` fetches historical 1-minute k-line data from the Binance API and stores it locally as monthly CSV files.
2.  **🛠️ Data Preparation**: `label.py` processes the raw data, engineers a variety of technical features (RSI, MACD, Bollinger Bands, etc.), and generates labels for supervised learning.
3.  **☁️ Model Training (AWS)**: `launch_sagemaker_job.py` uploads the prepared data to an S3 bucket and starts a training job on AWS SageMaker, utilizing GPU instances for speed.
4.  **🧠 Training Script**: `aws_train_model.py` is executed by SageMaker. It defines the PyTorch LSTM model, trains it with early stopping, and saves the model artifacts (`best_model.pth`, `scaler.pkl`) back to S3.
5.  **🚀 Model Deployment (AWS)**: After a successful training job, `launch_sagemaker_job.py` automatically deploys the best model to a SageMaker real-time endpoint. The `inference.py` script defines the logic for this endpoint.
6.  **📡 Live API Server**: `main.py` runs a local FastAPI server that loads the model artifacts and provides endpoints for inference.
7.  **💻 Client Interaction**: A frontend application can connect to the FastAPI server's WebSocket to receive live trading signals and display them on a dashboard.

## Features

- **Automated Data Fetching**: Pulls years of 1-minute ETH/USDT data from Binance.
- **Advanced Feature Engineering**: Creates 17+ technical indicators to feed the model.
- **Cloud-Based Training**: Leverages AWS SageMaker for scalable, powerful GPU-based model training.
- **Real-time Inference API**: A non-blocking API built with FastAPI to deliver signals with low latency.
- **Live Signal Streaming**: A WebSocket endpoint streams predictions every 5 seconds, perfect for a live dashboard.
- **Paper Trading Simulation**: Endpoints to run and monitor a simulated trading strategy in the background.
- **Backtesting Support**: `backtest.py` can generate signals over the entire historical dataset for strategy evaluation.

## Project Structure

```
.
├── eth_1m_data/              # Stores raw historical data from Binance
├── aws_train_model.py        # Defines the PyTorch model and the training logic for SageMaker.
├── backtest.py               # Generates signals over historical data for backtesting.
├── history.py                # Fetches historical 1-minute data from Binance.
├── inference.py              # Contains the code for the SageMaker real-time inference endpoint.
├── label.py                  # Performs feature engineering and creates labels for the dataset.
├── launch_sagemaker_job.py   # Orchestrates the AWS SageMaker training and deployment process.
├── main.py                   # The main FastAPI application for serving live signals.
├── paper_trade.py            # A script to run a simple paper trading simulation.
├── model_meta.json           # Stores model metadata, like the decision threshold.
├── requirements.txt          # A list of all Python dependencies for the project.
└── README.md                 # This documentation file.
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mawadSur/autopilot.git
    cd autopilot
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the root directory and add your Binance API keys. You only need keys for `history.py` if you are fetching data. For paper trading, you'll need testnet keys.
    ```env
    # For history.py (fetching data)
    BINANCE_KEY="your_binance_api_key"
    BINANCE_SECRET="your_binance_api_secret"
    ```

5.  **Configure AWS Credentials**:
    Ensure your environment is configured with AWS credentials. The simplest way is to use the AWS CLI:
    ```bash
    aws configure
    ```
    You will also need to **provide the correct IAM Role ARN** in `launch_sagemaker_job.py`.

## Workflow

Follow these steps to run the project from end to end:

1.  **Fetch Data**: Run `history.py` to download the historical data into the `eth_1m_data` folder. This may take a while.
    ```bash
    python history.py
    ```

2.  **Train and Deploy on AWS**: Execute the `launch_sagemaker_job.py` script. This will handle uploading data, training the model, and deploying it to a live endpoint.
    ```bash
    python launch_sagemaker_job.py
    ```
    After this step is complete, you will have a trained model in S3 and a live SageMaker endpoint.

3.  **Run the Live API Server**: To run the API locally, you must first download the model artifacts from S3.

    **How to Download Model Files from S3**

    After the SageMaker training job is complete, it saves a `model.tar.gz` file to your S3 bucket. You can find the exact path in the console output of the training job.

    Use the AWS CLI to download and extract the files:
    ```bash
    # Replace with the actual S3 path from your training job output
aws s3 cp s3://sagemaker-pytorch-2025-07-17-03-15-00-123/output/model.tar.gz .

    # This will extract best_model.pth and scaler.pkl into your current directory
    tar -xzvf model.tar.gz
    ```
    
    Now that the model files are local, start the FastAPI server:
    ```bash
    uvicorn main:app --reload
    ```


4.  **Connect a Frontend**: The API is now running on `http://127.0.0.1:8000`. You can connect a client to the WebSocket at `ws://127.0.0.1:8000/ws/signal-stream`.

## API Endpoints

The interactive API documentation (via Swagger UI) is available at `http://127.0.0.1:8000/docs` when the server is running.

- **`GET /`**: Welcome message.
- **`GET /signal/latest`**: Fetches the latest market data and returns a single "BUY" or "HOLD" signal.
- **`POST /papertrade/start`**: Starts a background paper trading simulation.
- **`GET /papertrade/status/{job_id}`**: Retrieves the status and log of a running simulation.
- **`WS /ws/signal-stream`**: WebSocket endpoint that streams the latest signal, confidence, and price every 5 seconds.

## Disclaimer

**This project is for educational purposes only and is not financial advice.** The predictions from this model are not guaranteed to be accurate. Trading cryptocurrencies involves significant risk, and you should never trade with money you cannot afford to lose. The author is not responsible for any financial losses incurred by using this software.
