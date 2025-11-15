from flask import Flask, request, jsonify,render_template
import os
import pdfplumber
import pandas as pd
import re
import requests
import time
from datetime import datetime
from openai import OpenAI


app = Flask(__name__)
client = OpenAI(api_key = "api_key")

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np


ASSETS = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "gold": "PAXGUSDT",
    "XRP" : "XRPUSDT",
    "Dogecoin" : "DOGEUSDT" 
}



def detect_fraud(X_data, y_data):
    """
    X_data: list of [income, expense] for previous transactions
    y_data: list of 0/1 labels for fraud
    new_tx: [income, expense] for new transaction
    """
    # Convert to numpy arrays
    X = np.array(X_data)
    y = np.array(y_data)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Scaling
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    # Model
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    # Prediction
    prediction = classifier.predict(X)

    return prediction[0] == 1


def get_historical_price(symbol, days_ago=7):
    url = "https://api.binance.com/api/v3/klines"
    end_time = int(time.time() * 1000)
    start_time = end_time - days_ago * 24 * 60 * 60 * 1000
    params = {
        "symbol": symbol,
        "interval": "1d",
        "startTime": start_time,
        "endTime": end_time,
        "limit": 2
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    if len(data) >= 2:
        today_price = float(data[-1][4])  # close price today
        prev_price = float(data[0][4])    # close price days_ago
        change = today_price - prev_price
        percent_change = (change / prev_price) * 100
        return today_price, prev_price, change, percent_change
    return None, None, None, None

def fetch_binance_ohlc(symbol, interval="1d", limit=2000):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    data = requests.get(url, params=params).json()

    ohlc = []
    for item in data:
        date = datetime.fromtimestamp(item[0]/1000).date()
        open_price = float(item[1])
        high_price = float(item[2])
        low_price = float(item[3])
        close_price = float(item[4])
        volume = float(item[5])
        ohlc.append([date, open_price, high_price, low_price, close_price, volume])

    df = pd.DataFrame(ohlc, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    return df
def create_trend_target(df, period=30):
    df = df.copy()
    df["Target"] = (df["Close"].shift(-1) - df["Close"].shift(-period)) > 0
    df["Target"] = df["Target"].astype(int)
    df.dropna(inplace=True)
    return df
def get_asset_trend(symbol):
    df = fetch_binance_ohlc(symbol)
    df = create_trend_target(df, period=30)

    # Son 30 günün ortalamaları
    df["Open_Mean30"] = df["Open"].rolling(30).mean()
    df["High_Mean30"] = df["High"].rolling(30).mean()
    df["Low_Mean30"] = df["Low"].rolling(30).mean()
    df["Close_Mean30"] = df["Close"].rolling(30).mean()
    df["Volume_Mean30"] = df["Volume"].rolling(30).mean()
    df.dropna(inplace=True)

    feature_cols = ["Open_Mean30", "High_Mean30", "Low_Mean30", "Close_Mean30", "Volume_Mean30"]
    X = df[feature_cols]
    y = df["Target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    today_features = X_scaled[-1].reshape(1, -1)
    prediction = model.predict(today_features)[0]

    result = {
        "symbol": symbol,
        "last_date": str(df['Date'].iloc[-1]),
        "last_close": df['Close'].iloc[-1],
        "trend": "Almaq olar" if prediction == 1 else "Almaq olmaz",
        "recent_data": df.tail(10).to_dict(orient='records')
    }
    return result


@app.route('/')
def index():
    return render_template("index.html")  # Frontend HTML faylı


@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'Fayl tapılmadı'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Fayl seçilməyib'}), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Zəhmət olmasa PDF faylı yükləyin'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # PDF-i oxu və əməliyyatları ayır
    with pdfplumber.open(filepath) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        # Available balance tap
        available_balance_match = re.search(r'Available balance\s+([\d.]+)', text)
        available_balance = float(available_balance_match.group(1)) if available_balance_match else 0.0

        # Sətirlərə böl
        lines = text.split('\n')
        transactions = []
        current_date = None
        current_debit = 0
        current_credit = 0
        current_balance = 0
        description_lines = []

        for line in lines:
            date_match = re.match(r'^(\d{4}-\d{2}-\d{2})\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*', line)
            if date_match:
                if current_date:
                    description = ' '.join(description_lines).strip()
                    if current_debit > 0:
                        transaction_type = "0"  # Debit
                        amount = current_debit
                    elif current_credit > 0:
                        transaction_type = "1"  # Credit
                        amount = current_credit
                    else:
                        transaction_type = "2"
                        amount = 0
                    transactions.append({
                        'Date': current_date,
                        'Debit': current_debit,
                        'Credit': current_credit,
                        'Balance': current_balance,
                        'Type': transaction_type,
                        'Amount': amount,
                        'Description': description
                    })
                # Yeni işləmi başla
                current_date = date_match.group(1)
                current_debit = float(date_match.group(2))
                current_credit = float(date_match.group(3))
                current_balance = float(date_match.group(4))
                description_lines = []
            elif current_date:
                if line.strip() and not re.match(r'^\d{4}-\d{2}-\d{2}', line):
                    description_lines.append(line.strip())

        # Son işləmi saxla
        if current_date:
            description = ' '.join(description_lines).strip()
            if current_debit > 0:
                transaction_type = "0"
                amount = current_debit
            elif current_credit > 0:
                transaction_type = "1"
                amount = current_credit
            else:
                transaction_type = "2"
                amount = 0
            transactions.append({
                'Date': current_date,
                'Debit': current_debit,
                'Credit': current_credit,
                'Balance': current_balance,
                'Type': transaction_type,
                'Amount': amount,
                'Description': description
            })

        # DataFrame
        df = pd.DataFrame(transactions)
        csv_path = os.path.join(UPLOAD_FOLDER, file.filename.replace('.pdf', '.csv'))
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')


        # Əsas statistika
        total_credit = df['Credit'].sum() if not df.empty else 0
        total_debit = df['Debit'].sum() if not df.empty else 0
        closing_balance = df['Balance'].iloc[-1] if not df.empty else available_balance
        opening_balance = df['Balance'].iloc[0] if not df.empty else available_balance
        transaction_count = len(df)

        # CSV və Excel faylları hazırla
        csv_path = os.path.join(UPLOAD_FOLDER, file.filename.replace('.pdf', '.csv'))
        excel_path = os.path.join(UPLOAD_FOLDER, file.filename.replace('.pdf', '.xlsx'))
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        df.to_excel(excel_path, index=False)

        data = {
            'filepath': filepath,
            'csvPath': csv_path,
            'excelPath': excel_path,
            'cardHolder': 'Unknown',
            'cardNumber': 'XXXX-XXXX-XXXX-XXXX',
            'currency': 'AZN',
            'period': f"{df['Date'].min() if not df.empty else '-'} - {df['Date'].max() if not df.empty else '-'}",
            'availableBalance': available_balance,
            'totalCredit': total_credit,
            'totalDebit': total_debit,
            'closingBalance': closing_balance,
            'openingBalance': opening_balance,
            'transactionCount': transaction_count,
            'autoDownloadCSV': True
        }

    return jsonify({'data': data})



@app.route("/api/analyze", methods = ['POST', 'GET'])
def analyze():
    # CSV fayllar
    csv_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith('.csv')]
    if not csv_files:
        return "CSV faylı tapılmadı!"

    # Son CSV
    csv_files.sort(key=lambda f: os.path.getmtime(os.path.join(UPLOAD_FOLDER, f)), reverse=True)
    latest_csv = os.path.join(UPLOAD_FOLDER, csv_files[0])
    df = pd.read_csv(latest_csv)
    df['Date'] = pd.to_datetime(df['Date'])

    # Income və expense
    X_data = df[['Credit', 'Debit']].values
    threshold = df['Debit'].mean() * 1.5

    y_data = (df['Debit'] > 1000).astype(int).values  # nümunə: Debit>1000 -> fraud
    df['Fraud'] = (df['Debit'] > threshold).astype(int)

    # Minimum iki class yoxlayırıq
    if len(set(y_data)) < 2:
        df['Fraud'] = 0
    else:
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=1)

        # Scale
        ss = StandardScaler()
        X_train = ss.fit_transform(X_train)
        X_scaled = ss.transform(X_data)

        # Model
        clf = LogisticRegression()
        clf.fit(X_train, y_train)

        # Proqnoz
        df['Fraud'] = clf.predict(X_scaled)
        print(df['Fraud'].head())
        df = df[df['Fraud'] != 0]   # <-- yalnız fraud əməliyyatları

    # Frontend üçün JSON
    transactions = []
    for i, row in df.iterrows():
        transactions.append({
            "date": row['Date'].strftime('%Y-%m-%d'),
            "credit": float(row['Credit']),
            "debit": float(row['Debit']),
            "balance": float(row['Balance']) if 'Balance' in df.columns else 0,
            "fraud": int(row['Fraud'])
        })




    symbols = {
        "Bitcoin": "BTCUSDT",
        "Ethereum": "ETHUSDT",
        "Gold": "PAXGUSDT",
        "XRP" : "XRPUSDT",
        "Dogecoin" : "DOGEUSDT",
        "Avalanche" : "AVAXUSDT",
        "NASDAQ Index" : "NASDAQUSDT"
    }
    
    results = {}
    for name, symbol in ASSETS.items():
        trend_data = get_asset_trend(symbol)

    stock_price = []
    for name, sym in symbols.items():
        try:
            today, week_ago, change, percent_change = get_historical_price(sym)
            stock_price.append({
                "name": name,
                "today": round(today, 2),
                "week_ago": round(week_ago, 2),
                "change": round(change, 2),
                "percent_change": round(percent_change, 2),
                "trend": trend_data['trend']
            })
        except Exception as e:
            print(f"Səhv: {name} ({sym}) → {e}")





    data = {"message": "Data hazırdır"}
    return render_template('recenttransactions.html', transactions=transactions, stock_price = stock_price,results = results, data=data)







@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "")
    print(data)

    if not message:
        return jsonify({"reply": "Zəhmət olmasa mesaj yazın."})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # və ya istədiyiniz modeli
            messages=[
                {"role": "system", "content": "Sən köməkçi chatbotsan.",
                "content": "Sən bir ekspert treydersən. Cavablarını qısa və dəqiq ver. İstifadəçiyə maliyyə və kripto mövzularında məsləhət ver."},
                {"role": "user", "content": message}
            ],
            max_tokens=150,
            temperature = 0.2
        )
        reply = response.choices[0].message.content
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"reply": f"Xəta baş verdi: {e}"}), 500






if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
