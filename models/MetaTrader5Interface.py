import MetaTrader5 as mt5
import pandas as pd

from datetime import datetime

from models.UITools import UITools
from models.CustomError import CustomError


def _request(symbol, order_type, volume=0.1, price=None, deviation=20):
    """
    Sends request to MetaTrader and returns its result
    :param symbol: symbol name
    :param order_type: must be one of mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY_STOP,
    mt5.ORDER_TYPE_BUY_STOP_LIMIT, mt5.ORDER_TYPE_SELL, mt5.ORDER_TYPE_SELL_LIMIT,mt5.ORDER_TYPE_SELL_STOP,
    mt5.ORDER_TYPE_SELL_STOP_LIMIT
    :param volume: volume to buy, sell or etc.
    :param price: price of symbol if not set will auto calculate it
    :param deviation: deviation of stock
    :return: result of transaction
    """
    if order_type not in (mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY_STOP,
                          mt5.ORDER_TYPE_BUY_STOP_LIMIT, mt5.ORDER_TYPE_SELL, mt5.ORDER_TYPE_SELL_LIMIT,
                          mt5.ORDER_TYPE_SELL_STOP, mt5.ORDER_TYPE_SELL_STOP_LIMIT):
        raise CustomError("Invalid order type!")

    point = mt5.symbol_info(symbol).point

    if price is None:
        price = mt5.symbol_info_tick(symbol).ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": price - 100 * point,
        "tp": price + 100 * point,
        "deviation": deviation,
        "magic": 234000,
        "comment": "python script open",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }

    return mt5.order_send(request)


def _evaluate_symbol(symbol):
    """
    Checks if symbol is valid and adds it to watch list
    :param symbol: symbol name
    :return: True if symbol is valid else false
    """
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        UITools.popup(f"{symbol} not found")
        mt5.shutdown()
        return False
    return True


class MetaTrader5Interface:

    @staticmethod
    def get_symbols(keyword):
        """
        Gets list of symbols that contain the keyword
        :param keyword: keyword to search with
        :return: list of symbols that contain the keyword or None if found nothing
        """
        if not mt5.initialize():
            UITools.popup(f"initialize() failed, error code = {mt5.last_error()}")
            return

        result = mt5.symbols_get(f"*{keyword}*")
        mt5.shutdown()
        return [symbol.name for symbol in result if symbol.currency_base == 'RLS']

    @staticmethod
    def get_symbol_data(symbol):
        """
        Gets price/volume history of the symbol over time
        :param symbol: symbol to check
        :return: pandas dataFrame
        """
        if not mt5.initialize():
            UITools.popup(f"initialize() failed, error code = {mt5.last_error()}")
            return

        if not _evaluate_symbol(symbol):
            UITools.popup(f"{symbol} is not valid! please check if its tradable")
            return

        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 5000)
        mt5.shutdown()

        stock_data = pd.DataFrame(rates)
        stock_data['time'] = pd.to_datetime(stock_data['time'], unit='s')
        return stock_data

    @staticmethod
    def buy_order(symbol, volume=0.1, price=None, deviation=20):
        """
        Send order to buy a symbol
        :param symbol: symbol name
        :param volume: volume to buy
        :param price: buy price if not set will auto calculate it
        :param deviation: deviation
        :return: True if trade transaction succeeded else False
        """
        if not mt5.initialize():
            UITools.popup(f"initialize() failed, error code = {mt5.last_error()}")
            return False

        if not _evaluate_symbol(symbol):
            UITools.popup(f"{symbol} is not valid! please check if its tradable")
            return False

        result = _request(symbol, mt5.ORDER_TYPE_BUY, volume, price, deviation)
        mt5.shutdown()
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            UITools.popup(f"order_send failed, return_code={result.retcode}")
            return False
        return True

    @staticmethod
    def sell_order(symbol, volume=0.1, price=None, deviation=20):
        """
        Send order to sell a symbol
        :param symbol: symbol name
        :param volume: volume to sell
        :param price: sell price if not set will auto calculate it
        :param deviation: deviation
        :return: True if trade transaction succeeded else False
        """
        if not mt5.initialize():
            UITools.popup(f"initialize() failed, error code = {mt5.last_error()}")
            return False

        if not _evaluate_symbol(symbol):
            UITools.popup(f"{symbol} is not valid! please check if its tradable")
            return False

        result = _request(symbol, mt5.ORDER_TYPE_SELL, volume, price, deviation)
        mt5.shutdown()
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            UITools.popup(f"order_send failed, return_code={result.retcode}")
            return False
        return True

    @staticmethod
    def get_symbol_current_orders(symbol):
        """
        Gets orders of the symbol in the last hour
        :param symbol: symbol to check
        :return: selected symbol trade history
        """
        if not mt5.initialize():
            UITools.popup(f"initialize() failed, error code = {mt5.last_error()}")
            return

        if not _evaluate_symbol(symbol):
            UITools.popup(f"{symbol} is not valid! please check if its tradable")
            return

        utc_from = datetime.today()
        ticks = mt5.copy_ticks_from(symbol, utc_from, 10000, mt5.COPY_TICKS_TRADE)

        stock_data = pd.DataFrame(ticks)
        stock_data['time'] = pd.to_datetime(stock_data['time'], unit='s')
        indexes = stock_data[stock_data['bid'] == 0].index
        stock_data = stock_data.drop(indexes)
        indexes = stock_data[stock_data['ask'] == 0].index
        stock_data = stock_data.drop(indexes)
        return stock_data

    @staticmethod
    def add_to_watch_list(symbol):
        """
        Adds selected symbol to mql5 watchlist so user can track its changes
        :param symbol: symbol to add
        :return: True if transaction is succeeded and False if it failed
        """
        if not _evaluate_symbol(symbol):
            UITools.popup(f"{symbol} is not valid! please check if its tradable")
            return False

        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                UITools.popup(f"symbol_select {symbol} failed, exit")
                mt5.shutdown()
                return False
        return True
