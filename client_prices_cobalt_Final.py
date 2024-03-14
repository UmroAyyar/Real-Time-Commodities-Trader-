import socket
import logging
import pandas as pd
import matplotlib.pyplot as plt
import json
from openai import OpenAI
from datetime import datetime, timedelta
pd.set_option('display.max_columns', None)
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)


class TradingClient:
    def __init__(self, host, port, openai_api_key):
        self.list_bid = []
        self.list_ask = []
        self.list_settlement_prices = []
        self.list_responses = []
        self.list_date = []
        self.buy_signal = []
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.listen_to_orders(host, port)
        #from sell
        # For Sell Signal Add
        self.initial_capital = float(10000.0)
        self.sell_signal = []
        self.in_position = False
        self.peak_price = 0
        self.buy_price = 0  ### grab from buy order dictionary
        self.shares_held = 0
        self.trailing_stop_loss_percentage = 0.15
        self.stop_loss_percentage = 0.15
        self.short_window = 10  # Short MA
        self.long_window = 20  # Long MA
        self.prices = []

    def listen_to_orders(self, host, port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((host, port))
                while True:
                    data = sock.recv(1024)
                    if not data:
                        break
                    received_data = data.decode('utf-8').strip()
                    if self.is_order_data(received_data):
                        order = self.parse_order(received_data)
                        logging.info('bid/ask')
                        self.add_order(order)
                        self.remove_old_orders()
                    else:
                        self.handle_tweet(received_data)

    # def is_order_data(self, data_str):
    #     return ',' in data_str and len(data_str.split(',')) == 6

    def is_order_data(self, data_str):
        if ',' in data_str and len(data_str.split(',')) == 6:
             order_id = data_str.split(',')[5].strip()
             return order_id.isdigit()  # Check if order ID consists only of digits
        return False

    def handle_tweet(self, tweet):
        # Handle tweet data
        try:
            completion = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system",
                    "content": "I'm analyzing tweets. I want you to reply 'yes' if the tweet below indicates an event resulting in significant loss of life, widespread devastation, or catastrophic impact impacting at least two hundred thousand people. Reply 'no' if the tweet is about any other topic."},
                    #"content": "I'm analyzing tweets. I want you to reply 'yes' if the tweet below indicates an event resulting in significant loss of life, widespread devastation, or catastrophic impact on hundreds of thousands of people. Reply 'no' if the tweet is about any other topic or does not meet this strict criteria."},
                    {"role": "user", "content": tweet}
                ]
            )
            response_content = completion.choices[0].message.content.strip().lower()

            # Update the list of responses for the latest date
            if len(self.list_date) > len(self.list_responses):
                self.list_responses.append(response_content)

            # Create a DataFrame to hold the responses
            response_df = pd.DataFrame({
                'date': self.list_date,
                'signal': self.list_responses
            })

            # Merge the response DataFrame with the existing DataFrame
            buy_df = pd.DataFrame({
                'date': self.list_date,
                'settlement_price': self.list_settlement_prices
            })

            buy_df = pd.merge(buy_df, response_df, on='date')
            buy_df['buy_signal'] = buy_df['signal'].apply(lambda x: 1 if x == 'yes' else 0)
            buy_df['qty'] = buy_df['buy_signal'].cumsum()
            # Calculate the 3-day moving average for the 'settlement_price'
            buy_df['short_ma'] = buy_df['settlement_price'].rolling(window=10).mean()
            # Calculate the 6-day moving average for the 'settlement_price'
            buy_df['long_ma'] = buy_df['settlement_price'].rolling(window=20).mean()
            buy_df['is_short_greater'] = (buy_df['short_ma'] > buy_df['long_ma']).astype(int)
            buy_df['ma_diff'] = buy_df['is_short_greater'].diff()
            buy_df['sell_signal1'] = (buy_df['ma_diff'] < 0).astype(int)
            buy_df['local_max'] = buy_df['settlement_price'].rolling(window=15).max()
            buy_df['trigger_2']= buy_df['local_max']*0.85
            buy_df['sell_signal2'] = (buy_df['settlement_price'] <= buy_df['trigger_2']).astype(int)
            buy_df['final_sell_signal'] = (buy_df['sell_signal1'] + buy_df['sell_signal2'] > 0).astype(int)
            #PNL coding
            selected_col = ['date', 'settlement_price', 'buy_signal', 'final_sell_signal']
            trade_df = buy_df.copy()
            trade_df = trade_df[selected_col]

            def calculate_master_signal(row):
                if row['buy_signal'] == 1 and row['final_sell_signal'] == 0:
                    return 1
                elif row['final_sell_signal'] == 1:
                    return -1
                else:
                    return 0
            # Apply the function to create the 'Master Signal' column
            trade_df['master_signal'] = trade_df.apply(calculate_master_signal, axis=1)

            def calculate_positions(row, prev_position):
                if row['master_signal'] == 1:
                    # If the current signal is a buy, increment position count
                    return prev_position+ 1
                elif row['master_signal'] == -1:
                    # If the current signal is a sell, reset position count to 0
                    return 0
                else:
                    # If the current signal is 0, maintain the previous position count
                    return prev_position

            # Initialize positions to 0
            prev_position = 0

            # Apply the function to create the 'Positions' column
            # trade_df['positions'] = df.apply(calculate_positions, axis=1)
            trade_df['positions'] = trade_df.apply(lambda row: calculate_positions(row, prev_position), axis=1)

            def calculate_total_positions(df):
                total_positions = []
                cumulative_count = 0
                last_zero_index = -1

                for index, row in df.iterrows():
                    if row['master_signal'] == -1:
                        cumulative_count = 0  # Reset cumulative count to 0
                    else:
                        cumulative_count += row['positions']
                    total_positions.append(cumulative_count)

                # Update the total_positions column in the DataFrame
                df['total_positions'] = total_positions
                return df

            # Call the function to calculate total_positions
            trade_df = calculate_total_positions(trade_df)

            def calculate_transacted_cash(df):
                transacted_cash = []
                prev_total_positions = None
                for index, row in df.iterrows():
                    if row['master_signal'] == 1:
                        transacted_cash.append(-1 * row['settlement_price'])
                    elif row['master_signal'] == -1:
                        if prev_total_positions is None:
                            transacted_cash.append(0)
                        else:
                            transacted_cash.append(prev_total_positions * row['settlement_price'])
                    else:
                        transacted_cash.append(0)
                    # Update prev_total_positions
                    prev_total_positions = row['total_positions']
                # Add transacted_cash column to the DataFrame
                df['transacted_cash'] = transacted_cash
                return df

            # Call the function to calculate transacted_cash
            trade_df = calculate_transacted_cash(trade_df)

            def calculate_final_cash(df, initial_capital=1000000):
                final_cash = []
                cumulative_cash = 0

                for cash_transacted in df['transacted_cash']:
                    cumulative_cash += cash_transacted
                    final_cash.append(initial_capital + cumulative_cash)

                # Add final_cash column to the DataFrame
                df['final_cash'] = final_cash

                return df

            # Call the function to calculate final_cash
            trade_df = calculate_final_cash(trade_df)

            def calculate_holdings_total_returns(df, initial_capital=1000000):
                holdings = df['settlement_price'] * df['total_positions']
                total_cash = df['final_cash']
                total = total_cash + holdings
                returns = ((total - initial_capital) / initial_capital)*100

                df['holdings'] = holdings
                df['total'] = total
                df['returns'] = returns
                df['returns'] = df['returns'].round(2)

                return df

            # Call the function to calculate holdings, total, and returns
            trade_df = calculate_holdings_total_returns(trade_df)

            def plot_assets(trade_df):
                print(trade_df)
                trade_df.plot(color='g', lw=.5)
                trade_df['holdings'].plot(color='g', lw=.5)
                trade_df['total_cash'].plot(color='r', lw=.5)
                trade_df['total'].plot(color='g', lw=.5)
                plt.title("Assets")
                plt.legend()
                plt.show()
                return trade_df


            # Example usage:
            # Assuming trade_df is already defined somewhere in your code
            plot_assets(trade_df)

        except Exception as e:
            logging.error("Error in handle_tweet: " + str(e))
            return pd.DataFrame()

    def parse_order(self, order_str):
        type, price, settlement_price, quantity, date, order_id = order_str.split(',')
        self.list_settlement_prices.append(float(settlement_price))
        self.list_date.append(date)
        return {
            'side': type.lower(),
            'price': float(price),
            'quantity': int(quantity),
            'date': date,
            'id': int(order_id)
        }

    def add_order(self, o):
        if o['side'] == 'bid':
            self.list_bid.append(o)
            self.list_bid.sort(key=lambda x: x['price'], reverse=True)
        elif o['side'] == 'ask':
            self.list_ask.append(o)
            self.list_ask.sort(key=lambda x: x['price'])
        else:
            raise Exception('Not a known side')

    def remove_old_orders(self):
        max_date = self.get_max_date()
        if max_date:
            cutoff_date = max_date - timedelta(days=4)
            self.list_bid = [o for o in self.list_bid if datetime.strptime(o['date'], '%Y-%m-%d') > cutoff_date]
            self.list_ask = [o for o in self.list_ask if datetime.strptime(o['date'], '%Y-%m-%d') > cutoff_date]

    def get_max_date(self):
        dates = [datetime.strptime(o['date'], '%Y-%m-%d') for o in self.list_bid + self.list_ask]
        max_date = max(dates) if dates else None
        return max_date

    # run update_price(self, TradingClient.self.list_settlement_prices[-1])

    # Sell Signal Code Start
    # keeps list 'prices' to calcuate the
    def calculate_local_max(data, window):
        local_max = data.rolling(window=window).max()
        # Shift by 1 to align local max with the corresponding data point
        local_max = local_max.shift(1)
        # Fill the first 'window - 1' elements with NaN (optional)
        local_max = local_max.fillna(method='fill')  # Forward fill with previous valid value
        return local_max

    def update_price(self, price):
        # Example usage with your DataFrame (assuming 'settlement_price' is the column)
        window_size = 15
        df['local_max'] = calculate_local_max(df['settlement_price'], window_size)

    def calculate_moving_average(data, window_size):
        if len(data) < window_size:
            return None
        return sum(data[-window_size:]) / window_size

    def add_moving_averages(df, window_short, window_long):
        df['short_ma'] = df['settlement_price'].rolling(window=window_short).apply(
            calculate_moving_average, raw=True)
        df['long_ma'] = df['settlement_price'].rolling(window=window_long).apply(
            calculate_moving_average, raw=True)
        return df

    def criss_cross(df):
        if df['short_ma'] > df['long_ma']:
            df['short_greater'] = 1
        else:
            df['short_greater'] = 0
        df['ma_diff'] = df['short_greater'].diff().fillna(0)

    def should_sell(self):
        threshold = 0.85  # 15% lower than local_max
        df['signal_1'] = (df['settlement_price'] <= df['local_max'] * threshold).astype(int)
        # df['signal_2'] = (df['settlement_price'] <= df['most_recent_purchase_price'] * threshold).astype(int)  ##### COME BACK TO THIS
        df['signal_3'] = (df['ma_diff'] < 0).astype(int)
        current_price = self.prices[-1] if self.prices else 0  ###EDIT REDUNDANT THING HERE. SHOULD SETTLEMENT_PRICES
        df['sell_signal'] = (df['signal_1'] + df['signal_3'] > 0).astype(int)
        print(df)

    def execute_sell(self):
        last_row = df.iloc[-1]
        if last_row['sell_signal'] > 0:
            sell_price = df['settlement_price']
            sell_quantity = self.shares_held
            sell_date  # Need to correct
            sell_order = {'side': 'sell', 'price': sell_price, 'quantity': sell_quantity,
                          'date': sell_date}  # Date needs to
            print("Executing Sell Order:", sell_order)
            # Append data to sell_df
            self.sell_df = self.sell_df.append({'date': sell_date, 'settlement_price': sell_price,
                                                'sell_signal': 1, 'qty': sell_quantity}, ignore_index=True)

        else:
            print("No Sell Order Executed")

    def print_order_lists(self):
        for order in self.list_bid:
            print(order)
        for order in self.list_ask:
            print(order)
        print()  # Blank line for separation
        return

    print('hello')

    def print_buy_signals(self):
        print(self.buy_signal)
        return

    def print_sell_signals(self):
        print(self.sell_df)
        return


HOST, PORT = "localhost", 9999
openai_api_key = "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"
ob = TradingClient(HOST, PORT, openai_api_key)



