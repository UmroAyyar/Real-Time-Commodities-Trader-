# install pandas
# install itertools
import socketserver
import time
import pandas as pd
from itertools import zip_longest

# Load data from CSV files
orders = pd.read_csv('orders.csv')
twitter_simulated = pd.read_csv('twitter_updated.csv')

class MyTCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        # Iterate over both DataFrames simultaneously
        for order, tweet in zip_longest(orders.iterrows(), twitter_simulated.iterrows(), fillvalue=(None, None)):
            if order[1] is not None:
                # Convert the order data to a string and send it
                _, order_data = order
                order_str = f"{order_data['type']},{order_data['price']},{order_data['settlement_price']},{order_data['quantity']},{order_data['date']},{order_data['order_id']}\n"
                self.request.sendall(order_str.encode('utf-8'))
                time.sleep(1)

            if tweet[1] is not None:
                # Convert the tweet data to a string and send it
                _, tweet_data = tweet
                message = f"{tweet_data['text']}\n"
                self.request.sendall(message.encode('utf-8'))
                time.sleep(1)

if __name__ == "__main__":
    HOST, PORT = "localhost", 9999
    with socketserver.TCPServer((HOST, PORT), MyTCPHandler) as server:
        server.serve_forever()
