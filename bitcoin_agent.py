testowa linia

import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import json
import logging
from datetime import datetime
import time
import requests
import os
import shutil
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Konfiguracja logowania
logging.basicConfig(filename='bitcoin_agent.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Inicjalizacja analizatora sentymentu
analyzer = SentimentIntensityAnalyzer()

# Klasa portfela (wirtualny portfel)
class VirtualWallet:
    def __init__(self, initial_cash=10000):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.btc = 0
        self.transactions = []
        self.last_trade_time = 0  # Timestamp ostatniej transakcji
        self.average_buy_price = 0  # Średnia cena kupna BTC
        self.highest_price_since_buy = 0  # Najwyższa cena od ostatniego kupna
        self.total_realized_profit = 0  # Całkowity zrealizowany zysk

    def buy(self, price, amount):
        cost = price * amount
        current_time = time.time()
        logging.info(f"Próba kupna: {amount} BTC po cenie {price}, koszt: {cost}, dostępna gotówka: {self.cash}")
        
        # Ograniczenie częstotliwości transakcji (co najmniej 15 minut między transakcjami)
        if current_time - self.last_trade_time < 900:  # 15 minut = 900 sekund
            logging.warning("Zbyt krótki czas od ostatniej transakcji, pomijam kupno")
            return False
        
        # Ograniczenie maksymalnej wielkości transakcji (max 5% początkowego kapitału na transakcję)
        max_cost = 0.05 * self.initial_cash
        if cost > max_cost:
            logging.warning(f"Koszt transakcji ({cost}) przekracza maksymalny limit ({max_cost}), pomijam kupno")
            return False
        
        if cost <= self.cash:
            self.cash -= cost
            old_btc = self.btc
            self.btc += amount
            # Oblicz średnią cenę kupna
            if old_btc == 0:
                self.average_buy_price = price
            else:
                self.average_buy_price = (self.average_buy_price * old_btc + price * amount) / self.btc
            self.highest_price_since_buy = price
            self.transactions.append({
                'type': 'buy', 
                'price': price, 
                'amount': amount, 
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            self.last_trade_time = current_time
            logging.info(f"Kupiono {amount} BTC po {price}, gotówka: {self.cash}, średnia cena kupna: {self.average_buy_price}")
            self.save_state()
            return True
        logging.warning(f"Niewystarczające środki do kupna: {self.cash} < {cost}")
        return False

    def sell(self, price, amount):
        current_time = time.time()
        logging.info(f"Próba sprzedaży: {amount} BTC po cenie {price}, dostępne BTC: {self.btc}")
        
        # Ograniczenie częstotliwości transakcji (co najmniej 15 minut między transakcjami)
        if current_time - self.last_trade_time < 900:  # 15 minut = 900 sekund
            logging.warning("Zbyt krótki czas od ostatniej transakcji, pomijam sprzedaż")
            return False
        
        if amount <= self.btc:
            self.cash += price * amount
            profit = (price - self.average_buy_price) * amount
            self.total_realized_profit += profit
            self.btc -= amount
            if self.btc == 0:
                self.average_buy_price = 0
                self.highest_price_since_buy = 0
            else:
                # Aktualizuj średnią cenę kupna (dla uproszczenia zakładamy, że sprzedajemy po średniej cenie)
                self.average_buy_price = self.average_buy_price  # Możemy dodać bardziej zaawansowaną logikę
            self.transactions.append({
                'type': 'sell', 
                'price': price, 
                'amount': amount, 
                'profit': profit,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            self.last_trade_time = current_time
            logging.info(f"Sprzedano {amount} BTC po {price}, gotówka: {self.cash}, zysk z transakcji: {profit}, całkowity zrealizowany zysk: {self.total_realized_profit}")
            self.save_state()
            return True
        logging.warning(f"Niewystarczające BTC do sprzedaży: {self.btc} < {amount}")
        return False

    def get_value(self, current_price):
        return self.cash + self.btc * current_price

    def get_unrealized_profit(self, current_price):
        if self.btc > 0:
            return (current_price - self.average_buy_price) * self.btc
        return 0

    def save_state(self):
        try:
            wallet_data = {
                'cash': self.cash,
                'btc': self.btc,
                'transactions': self.transactions,
                'last_trade_time': self.last_trade_time,
                'average_buy_price': self.average_buy_price,
                'highest_price_since_buy': self.highest_price_since_buy,
                'total_realized_profit': self.total_realized_profit
            }
            logging.info(f"Zapisywanie portfela: gotówka {self.cash}, BTC {self.btc}, transakcje {len(self.transactions)}")
            temp_file = '/home/marcin/wallet.json.tmp'
            with open(temp_file, 'w') as f:
                json.dump(wallet_data, f)
            shutil.move(temp_file, '/home/marcin/wallet.json')
            logging.info("Portfel zapisany pomyślnie")
        except Exception as e:
            logging.error(f"Błąd zapisu portfela: {e}")
            raise

    @staticmethod
    def load_state():
        try:
            if not os.path.exists('/home/marcin/wallet.json'):
                logging.warning("Brak pliku wallet.json, tworzę nowy portfel")
                wallet = VirtualWallet()
                wallet.save_state()
                return wallet
            with open('/home/marcin/wallet.json', 'r') as f:
                data = json.load(f)
                wallet = VirtualWallet(0)
                wallet.cash = data['cash']
                wallet.btc = data['btc']
                wallet.transactions = data['transactions']
                wallet.last_trade_time = data.get('last_trade_time', 0)
                wallet.average_buy_price = data.get('average_buy_price', 0)
                wallet.highest_price_since_buy = data.get('highest_price_since_buy', 0)
                wallet.total_realized_profit = data.get('total_realized_profit', 0)
                wallet.initial_cash = 10000  # Ustawiamy początkowy kapitał
                logging.info(f"Wczytano portfel: gotówka {wallet.cash}, BTC {wallet.btc}, transakcje {len(wallet.transactions)}, całkowity zrealizowany zysk: {wallet.total_realized_profit}")
                return wallet
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error(f"Krytyczny błąd odczytu portfela: {e}, tworzę nowy portfel")
            wallet = VirtualWallet()
            wallet.save_state()
            return wallet

# Pobieranie danych z giełdy z obliczaniem wskaźników technicznych
def fetch_data(exchange, symbol='BTC/USDT', timeframe='1h', limit=500):
    logging.info(f"Pobieranie danych: {symbol}, timeframe: {timeframe}, limit: {limit}")
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Oblicz RSI (14 okresów)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Oblicz MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Oblicz średnie kroczące (SMA 20 i 50)
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Oblicz zmienność (standardowe odchylenie ceny zamknięcia)
        df['volatility'] = df['close'].rolling(window=20).std()
        
        logging.info(f"Pobrano dane: {len(df)} rekordów")
        return df
    except Exception as e:
        logging.error(f"Błąd pobierania danych z giełdy: {e}")
        raise

# Pobieranie danych z CoinGecko API
def fetch_coingecko_data():
    logging.info("Pobieranie danych z CoinGecko API")
    try:
        headers = {
            'x-cg-demo-api-key': 'CG-FfTvL85iVjP5Cf1DrEdaY9Ct'
        }
        url = "https://api.coingecko.com/api/v3/coins/bitcoin"
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"Błąd API CoinGecko: Status {response.status_code}")
        
        latest_data = response.json()
        market_cap = latest_data['market_data']['market_cap']['usd']
        volume_24h = latest_data['market_data']['total_volume']['usd']
        
        logging.info(f"Wolumen 24h (USD): {volume_24h}, Kapitalizacja rynkowa (USD): {market_cap}")
        return volume_24h, market_cap
    except Exception as e:
        logging.error(f"Błąd pobierania danych z CoinGecko API: {e}")
        return 0, 0

# Pobieranie krótkoterminowych danych do wykrywania nagłych spadków
def fetch_short_term_data(exchange, symbol='BTC/USDT', timeframe='5m', limit=12):
    logging.info(f"Pobieranie krótkoterminowych danych: {symbol}, timeframe: {timeframe}, limit: {limit}")
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        logging.info(f"Pobrano krótkoterminowe dane: {len(df)} rekordów")
        return df
    except Exception as e:
        logging.error(f"Błąd pobierania krótkoterminowych danych: {e}")
        raise

# Analiza sezonowości i trendów historycznych
def fetch_historical_data(exchange, symbol='BTC/USDT', timeframe='1d', since=None):
    logging.info("Pobieranie danych historycznych do analizy sezonowości")
    try:
        since = exchange.parse8601('2020-01-01T00:00:00Z') if since is None else since
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Dodaj kolumnę z miesiącem i dniem dla analizy sezonowości
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        
        # Oblicz średnie zmiany cen w danym miesiącu/dniu
        df['price_change'] = df['close'].pct_change()
        seasonal_trends = df.groupby(['month', 'day'])['price_change'].mean().to_dict()
        
        logging.info("Zakończono analizę sezonowości")
        return seasonal_trends
    except Exception as e:
        logging.error(f"Błąd pobierania danych historycznych: {e}")
        raise

# Pobieranie i analiza sentymentu z NewsAPI
def fetch_sentiment():
    logging.info("Rozpoczynam pobieranie sentymentu z NewsAPI")
    try:
        url = "https://newsapi.org/v2/everything?q=bitcoin&apiKey=de4bf7efef4547b49e856ed53f44829d"
        response = requests.get(url)
        logging.info(f"Status HTTP: {response.status_code}, Odpowiedź: {response.text}")
        
        if response.status_code != 200:
            raise Exception(f"Błąd API: Status {response.status_code}")
        
        articles = response.json()['articles']
        posts = [{'text': article['title']} for article in articles]
        
        sentiments = []
        for post in posts:
            score = analyzer.polarity_scores(post['text'])
            sentiments.append(score['compound'])
        
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        logging.info(f"Średni sentyment z NewsAPI: {avg_sentiment}")
        return avg_sentiment
    except Exception as e:
        logging.error(f"Błąd analizy sentymentu, używam danych testowych: {e}")
        posts = [
            {"text": "Bitcoin to the moon! 🚀 #BTC"},
            {"text": "BTC crashing, sell now! 😱"},
            {"text": "Just bought some Bitcoin, feeling good!"}
        ]
        sentiments = []
        for post in posts:
            score = analyzer.polarity_scores(post['text'])
            sentiments.append(score['compound'])
        
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        logging.info(f"Średni sentyment z X: {avg_sentiment}")
        return avg_sentiment

# Zapisz dane historyczne do pliku
def save_history_data(history_data):
    try:
        temp_file = '/home/marcin/history_data.json.tmp'
        with open(temp_file, 'w') as f:
            json.dump(history_data, f)
        shutil.move(temp_file, '/home/marcin/history_data.json')
        logging.info("Zapisano dane historyczne")
    except Exception as e:
        logging.error(f"Błąd zapisu danych historycznych: {e}")

# Wczytaj dane historyczne z pliku
def load_history_data():
    try:
        if os.path.exists('/home/marcin/history_data.json'):
            with open('/home/marcin/history_data.json', 'r') as f:
                data = json.load(f)
                logging.info(f"Wczytano dane historyczne: {len(data)} wpisów")
                return data
        logging.warning("Brak pliku history_data.json, zwracam pustą listę")
        return []
    except Exception as e:
        logging.error(f"Błąd wczytywania danych historycznych: {e}")
        return []

# Optymalizacja progów kupna i sprzedaży
def optimize_thresholds(history_data, model, current_data):
    logging.info(f"Optymalizacja progów: liczba wpisów w history_data: {len(history_data)}")
    
    # Użyj tylko danych z history_data, które mają przewidywane zwroty
    if len(history_data) < 5:  # Minimalna liczba danych do optymalizacji
        logging.warning(f"Za mało danych historycznych do optymalizacji progów: {len(history_data)} wpisów, używam domyślnych wartości")
        return 0.1, -0.1
    
    # Oblicz statystyki predicted_return dla history_data
    predicted_returns = [entry['predicted_return'] for entry in history_data]
    return_mean = np.mean(predicted_returns)
    return_std = np.std(predicted_returns) if len(predicted_returns) > 1 else 0.0001
    logging.info(f"Statystyki predicted_return: średnia: {return_mean}, odchylenie standardowe: {return_std}")
    
    # Oblicz zmienność i RSI z current_data
    volatility = current_data['volatility'].iloc[-1] if 'volatility' in current_data else 0
    rsi = current_data['rsi'].iloc[-1]
    
    # Dostosuj progi w zależności od zmienności i RSI
    volatility_factor = 1 + volatility / current_data['close'].iloc[-1]  # Normalizujemy zmienność względem ceny
    rsi_factor = 1 if 30 < rsi < 70 else (1.5 if rsi <= 30 else 0.5)  # Zwiększamy progi przy wyprzedaniu, zmniejszamy przy wykupieniu
    
    # Zakresy progów do przetestowania
    buy_thresholds = np.linspace(-0.1 * volatility_factor * rsi_factor, 0.5 * volatility_factor * rsi_factor, 13)
    sell_thresholds = np.linspace(-0.5 * volatility_factor * rsi_factor, 0.1 * volatility_factor * rsi_factor, 13)
    
    best_profit = float('-inf')
    best_buy_threshold = 0.1
    best_sell_threshold = -0.1
    
    for buy_threshold in buy_thresholds:
        for sell_threshold in sell_thresholds:
            # Symulacja portfela
            wallet_sim = VirtualWallet(initial_cash=10000)
            buy_count = 0
            sell_count = 0
            for entry in history_data:
                predicted_return = entry['predicted_return']
                sentiment = entry['sentiment']
                price = entry['price']
                
                # Normalizacja predicted_return i bardziej elastyczne warunki
                normalized_return = (predicted_return - return_mean) / return_std if return_std != 0 else predicted_return
                if normalized_return > 0.002 and sentiment > buy_threshold:  # Zmniejszamy próg dla kupna
                    if wallet_sim.buy(price, 0.01):
                        buy_count += 1
                elif normalized_return < -0.002 and sentiment < sell_threshold and wallet_sim.btc >= 0.01:  # Zmniejszamy próg dla sprzedaży
                    if wallet_sim.sell(price, 0.01):
                        sell_count += 1
            
            # Oblicz zysk
            final_value = wallet_sim.get_value(price)
            profit = final_value - 10000
            
            logging.debug(f"Symulacja dla progów: Kupno {buy_threshold}, Sprzedaż {sell_threshold}, Kupno: {buy_count}, Sprzedaż: {sell_count}, Zysk: {profit}")
            
            if profit > best_profit:
                best_profit = profit
                best_buy_threshold = buy_threshold
                best_sell_threshold = sell_threshold
    
    logging.info(f"Najlepsze progi: Kupno {best_buy_threshold}, Sprzedaż {best_sell_threshold}, Zysk: {best_profit}")
    return best_buy_threshold, best_sell_threshold

# Trenowanie modelu z uwzględnieniem wskaźników technicznych i danych z CoinGecko
def train_model(data, sentiment, seasonal_trend, volume_24h, market_cap):
    try:
        data['returns'] = data['close'].pct_change()
        data['lag1'] = data['close'].shift(1)
        data['lag2'] = data['close'].shift(2)
        data['volume_change'] = data['volume'].pct_change()
        data['sentiment'] = sentiment
        data['seasonal_trend'] = seasonal_trend
        data['volume_24h'] = volume_24h
        data['market_cap'] = market_cap
        data = data.dropna()
        
        X = data[['lag1', 'lag2', 'volume_change', 'rsi', 'macd', 'macd_signal', 'sma_20', 'sma_50', 'sentiment', 'seasonal_trend', 'volume_24h', 'market_cap']]
        y = data['returns']
        
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X, y)
        return model, data
    except Exception as e:
        logging.error(f"Błąd trenowania modelu: {e}")
        raise

# Predykcja
def predict(model, data):
    try:
        latest = data.tail(1)[['lag1', 'lag2', 'volume_change', 'rsi', 'macd', 'macd_signal', 'sma_20', 'sma_50', 'sentiment', 'seasonal_trend', 'volume_24h', 'market_cap']]
        return model.predict(latest)[0]
    except Exception as e:
        logging.error(f"Błąd predykcji: {e}")
        raise

# Wczytaj progi sentymentu i interwał
def load_config():
    CONFIG_FILE = "/home/marcin/config.json"
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                buy_threshold = config.get("buy_threshold", 0.1)
                sell_threshold = config.get("sell_threshold", -0.1)
                interval = config.get("interval", 300)
                logging.info(f"Wczytano konfigurację: Kupno {buy_threshold}, Sprzedaż {sell_threshold}, Interwał: {interval} sekund")
                return float(buy_threshold), float(sell_threshold), int(interval)
        else:
            logging.warning("Brak pliku konfiguracyjnego, używam domyślnych wartości")
            return 0.1, -0.1, 300
    except Exception as e:
        logging.error(f"Błąd wczytywania konfiguracji: {e}, używam domyślnych wartości")
        return 0.1, -0.1, 300

# Zapisz konfigurację
def save_config(buy_threshold, sell_threshold, interval):
    CONFIG_FILE = "/home/marcin/config.json"
    config = {
        "buy_threshold": float(buy_threshold),
        "sell_threshold": float(sell_threshold),
        "interval": int(interval)
    }
    try:
        temp_file = '/home/marcin/config.json.tmp'
        with open(temp_file, 'w') as f:
            json.dump(config, f)
        shutil.move(temp_file, CONFIG_FILE)
        logging.info(f"Konfiguracja zapisana: {config}")
    except Exception as e:
        logging.error(f"Błąd zapisu konfiguracji: {e}")

# Oblicz optymizm i dynamiczną wielkość transakcji
def calculate_trade_amount(sentiment, predicted_return, base_amount=0.01):
    try:
        optimism = (sentiment + predicted_return * 100) / 2
        max_multiplier = 5  # Ograniczamy maksymalny mnożnik do 5, aby zmniejszyć ryzyko
        multiplier = min(max(optimism * 2, 1), max_multiplier)
        trade_amount = base_amount * multiplier
        return trade_amount
    except Exception as e:
        logging.error(f"Błąd obliczania trade_amount: {e}")
        return base_amount

# Wykryj nagły spadek
def detect_price_drop(short_term_data, threshold=-0.05):
    try:
        if len(short_term_data) < 2:
            return False
        price_change = (short_term_data['close'].iloc[-1] - short_term_data['close'].iloc[0]) / short_term_data['close'].iloc[0]
        return price_change < threshold
    except Exception as e:
        logging.error(f"Błąd wykrywania nagłego spadku: {e}")
        return False

# Główna pętla agenta
def run_agent():
    try:
        logging.info("Rozpoczynam działanie agenta")
        exchange = ccxt.binance()
        symbol = 'BTC/USDT'
        
        # Pobierz dane historyczne do analizy sezonowości
        seasonal_trends = fetch_historical_data(exchange, symbol)
        
        # Wczytaj dane historyczne z pliku
        history_data = load_history_data()
        
        # Licznik iteracji do okresowej optymalizacji progów
        iteration_count = 0
        optimization_interval = 1  # Optymalizuj progi co 1 iterację (dla testów)
        
        while True:
            try:
                # Wczytaj portfel na początku każdej iteracji
                wallet = VirtualWallet.load_state()
                
                # Pobierz dane
                data = fetch_data(exchange, symbol)
                current_price = data['close'].iloc[-1]
                
                # Pobierz krótkoterminowe dane do wykrywania spadków
                short_term_data = fetch_short_term_data(exchange, symbol)
                
                # Pobierz sentyment
                sentiment = fetch_sentiment()
                
                # Pobierz dane z CoinGecko
                volume_24h, market_cap = fetch_coingecko_data()
                
                # Przygotuj dane do modelu
                data['lag1'] = data['close'].shift(1)
                data['lag2'] = data['close'].shift(2)
                data['volume_change'] = data['volume'].pct_change()
                
                # Oblicz trend sezonowy dla bieżącej daty
                current_date = datetime.now()
                month, day = current_date.month, current_date.day
                seasonal_trend = seasonal_trends.get((month, day), 0)
                
                # Trenuj model z sentymentem, trendem sezonowym i danymi z CoinGecko
                model, data = train_model(data, sentiment, seasonal_trend, volume_24h, market_cap)
                
                # Loguj cechy modelu
                logging.info(f"Cechy modelu: {data.tail(1)[['lag1', 'lag2', 'volume_change', 'rsi', 'macd', 'macd_signal', 'sma_20', 'sma_50', 'sentiment', 'seasonal_trend', 'volume_24h', 'market_cap']].to_dict()}")
                
                # Przewiduj zwrot
                predicted_return = predict(model, data)
                logging.info(f"Przewidywany zwrot: {predicted_return}, Sentyment: {sentiment}, Trend sezonowy: {seasonal_trend}")
                
                # Wczytaj progi sentymentu i interwał
                buy_threshold, sell_threshold, interval = load_config()
                logging.info(f"Progi: Kupno {buy_threshold}, Sprzedaż {sell_threshold}, Interwał: {interval} sekund")
                
                # Zapisz dane do optymalizacji progów
                history_data.append({
                    'predicted_return': predicted_return,
                    'sentiment': sentiment,
                    'price': current_price
                })
                
                # Oblicz wartość portfela i proporcje
                portfolio_value = wallet.get_value(current_price)
                btc_value = wallet.btc * current_price
                btc_ratio = btc_value / portfolio_value if portfolio_value > 0 else 0
                cash_ratio = wallet.cash / wallet.initial_cash
                unrealized_profit = wallet.get_unrealized_profit(current_price)
                total_profit = wallet.total_realized_profit + unrealized_profit
                
                logging.info(f"Portfel: Wartość: {portfolio_value} USD, Gotówka: {wallet.cash} USD, BTC: {wallet.btc}, Proporcja BTC: {btc_ratio:.2f}, Proporcja gotówki: {cash_ratio:.2f}, Zrealizowany zysk: {wallet.total_realized_profit:.2f} USD, Niezrealizowany zysk: {unrealized_profit:.2f} USD, Całkowity zysk: {total_profit:.2f} USD")
                
                # Sprawdź, czy portfel osiągnął limit strat/zysków
                if total_profit < -0.2 * wallet.initial_cash:  # 20% straty
                    logging.warning("Portfel osiągnął limit strat (-20%), zatrzymuję handel")
                    break
                if total_profit > 0.5 * wallet.initial_cash:  # 50% zysku
                    logging.info("Portfel osiągnął limit zysków (+50%), zatrzymuję handel")
                    break
                
                # Sprawdź, czy portfel jest w stanie krytycznym
                if wallet.cash < 0.05 * wallet.initial_cash and wallet.btc == 0:
                    logging.warning("Portfel w stanie krytycznym: brak gotówki i BTC, zatrzymuję handel")
                    break
                
                # Wykryj nagły spadek
                if detect_price_drop(short_term_data):
                    logging.info(f"Wykryto nagły spadek ceny! Sprzedaję cały portfel: {wallet.btc} BTC")
                    if wallet.btc > 0:
                        wallet.sell(current_price, wallet.btc)
                else:
                    # Oblicz dynamiczną wielkość transakcji na podstawie optymizmu
                    trade_amount = calculate_trade_amount(sentiment, predicted_return)
                    logging.info(f"Dynamiczna wielkość transakcji: {trade_amount} BTC")
                    
                    # Logika handlu
                    cost = trade_amount * current_price
                    should_buy = predicted_return > 0.005 and sentiment > buy_threshold and wallet.cash > cost and cash_ratio > 0.1  # Zaostrzamy warunek kupna
                    should_sell = predicted_return < -0.005 and sentiment < sell_threshold and wallet.btc >= trade_amount  # Zaostrzamy warunek sprzedaży
                    
                    # Stop-loss: sprzedaj, jeśli cena spadła o 5% poniżej średniej ceny kupna
                    if wallet.btc > 0 and current_price < wallet.average_buy_price * 0.95:
                        should_sell = True
                        logging.info(f"Stop-loss aktywowany: cena ({current_price}) spadła poniżej 95% średniej ceny kupna ({wallet.average_buy_price})")
                    
                    # Trailing stop: sprzedaj, jeśli cena spadła o 3% od najwyższej ceny od kupna
                    if wallet.btc > 0:
                        wallet.highest_price_since_buy = max(wallet.highest_price_since_buy, current_price)
                        if current_price < wallet.highest_price_since_buy * 0.97:
                            should_sell = True
                            logging.info(f"Trailing stop aktywowany: cena ({current_price}) spadła poniżej 97% najwyższej ceny ({wallet.highest_price_since_buy})")
                    
                    # Dodatkowe warunki zarządzania portfelem
                    if btc_ratio > 0.8 and predicted_return < 0:  # Sprzedaj, jeśli portfel jest zdominowany przez BTC i rynek spada
                        should_sell = True
                        logging.info("Portfel zdominowany przez BTC i rynek spada, aktywuję sprzedaż")
                    
                    if should_buy:
                        wallet.buy(current_price, trade_amount)
                        logging.info(f"Transakcja: Kupiono {trade_amount} BTC przy zwrocie {predicted_return} i sentymencie {sentiment}")
                    elif should_sell:
                        wallet.sell(current_price, trade_amount)
                        logging.info(f"Transakcja: Sprzedano {trade_amount} BTC przy zwrocie {predicted_return} i sentymencie {sentiment}")
                    else:
                        logging.info("Brak transakcji: warunki kupna/sprzedaży nie zostały spełnione")
                
                # Zapisz dane do analizy
                portfolio_value = wallet.get_value(current_price)
                with open('/home/marcin/portfolio_history.json', 'a') as f:
                    json.dump({
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'value': portfolio_value,
                        'price': current_price,
                        'sentiment': sentiment,
                        'predicted_return': predicted_return
                    }, f)
                    f.write('\n')
                
                # Analiza korelacji
                correlation = data['close'].corr(data['volume'])
                logging.info(f"Wartość portfela: {portfolio_value} USD, cena BTC: {current_price}, korelacja: {correlation}, sentyment: {sentiment}")
                
                # Okresowa optymalizacja progów
                iteration_count += 1
                if iteration_count % optimization_interval == 0:
                    logging.info("Optymalizacja progów kupna i sprzedaży")
                    new_buy_threshold, new_sell_threshold = optimize_thresholds(history_data, model, data)
                    save_config(new_buy_threshold, new_sell_threshold, interval)
                    buy_threshold, sell_threshold = new_buy_threshold, new_sell_threshold
                    logging.info(f"Zaktualizowano progi: Kupno {buy_threshold}, Sprzedaż {sell_threshold}")
                
                # Zapisz dane historyczne
                save_history_data(history_data)
                
                time.sleep(interval)
                
            except Exception as e:
                logging.error(f"Błąd w pętli agenta: {e}")
                time.sleep(60)
                
    except Exception as e:
        logging.error(f"Krytyczny błąd agenta: {e}")

if __name__ == '__main__':
    run_agent()
