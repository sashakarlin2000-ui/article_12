import yfinance as yf
import pandas as pd

# Список тикеров по твоему ТЗ
TICKERS = ["^GSPC", "AAPL", "MSFT", "NVDA", "AMZN", "TSLA"]

def fetch_market_data(tickers=TICKERS, period="10y"):
    """
    Скачивает дневные исторические данные по списку тикеров.
    """
    all_data = []
    
    for ticker in tickers:
        print(f"Скачиваю данные для: {ticker}...")
        
        # Скачиваем данные с интервалом 1 день
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=True)
        
        if df.empty:
            print(f"Предупреждение: по тикеру {ticker} ничего не скачалось!")
            continue
            
        # Красивое имя для индекса S&P 500
        ticker_name = "SPX" if ticker == "^GSPC" else ticker
        
        # Убираем мультииндекс в колонках, если yfinance его вернул
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.reset_index()
        
        # Оставляем только нужные колонки по ТЗ
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = df[[col for col in required_cols if col in df.columns]].copy()
        
        # Добавляем колонку с именем тикера, чтобы разделять акции в общем датасете
        df['Ticker'] = ticker_name
        
        all_data.append(df)
        
    if not all_data:
        raise ValueError("Не удалось скачать данные ни по одному тикеру!")
        
    # Объединяем всё в один большой датафрейм
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Приводим дату к чистому формату без временной зоны
    combined_df['Date'] = pd.to_datetime(combined_df['Date']).dt.tz_localize(None)
    
    return combined_df

if __name__ == "__main__":
    # Тестовый запуск скачивания
    raw_df = fetch_market_data()
    print("\nСкачивание завершено успешно!")
    print(raw_df.head())
    print(f"Всего строк в сырых данных: {len(raw_df)}")