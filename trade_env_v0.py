import os
import zipfile
import random
from typing import Dict, Optional

import pandas as pd
import numpy as np
from scipy.stats import entropy, differential_entropy
import talib as ta

import gymnasium as gym
from gymnasium import spaces


class Config:
    def __init__(self):
        self.data_dir = 'data'
        self.file_name = 'BTC_USDT-30m.feather'
        self.file_path = os.path.join(self.data_dir, self.file_name)
        self.max_stakes = 5
        self.initial_balance = 1000
        self.buy_fee = 0.0005
        self.sell_fee = 0.0005
        self.window_size = 144
        self.render_mode = 'print'
        self.mode = 'train'

class TradingEnv(gym.Env):
    def __init__(self, file_path, max_stakes=5, initial_balance=1000, buy_fee=0.0005, sell_fee=0.0005):       
        super(TradingEnv, self).__init__()
        self.current_step = -1

        # Environment parameters
        self.timeframe = '30m'
        self.base_currency = 'BTC'
        self.quote_currency = 'USDT'
        self.buy_fee = buy_fee
        self.sell_fee = sell_fee

        # list of indicators to use in observation space
        self.indicators = ['normalized_close', 'normalized_volume', 'ema5', 'ema13', 'ema21', 'sma50', 'sma100']
        self.data = load_data(file_path)

        # Set up environment variables
        self.max_stakes = max_stakes if max_stakes > 0 else 1
        self.stakes: Dict[int, Optional[dict]] = {i: None for i in range(0, max_stakes)}

        self.initial_balance = initial_balance
        self.quote_balance = initial_balance
        self.base_balance = 0
        self.base_currency_balance = self.base_balance * self.data['close'].values[self.current_step]
        self.total_balance = self.quote_balance + self.base_currency_balance
        self.balance_fraction = self.quote_balance / self.total_balance

        # Observation space: [closing price, MAs, balance fraction,]
        num_obs_features = len(self.indicators) + 1 + (self.max_stakes*2)  # indicators + balance + per-stake information
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_obs_features,), dtype=np.float32)
        
        # Action space: [sell/hold decisions for each stake (continuous, thresholded) + buy fraction]
        self.action_space = spaces.Box(low=0, high=1, shape=(self.max_stakes + 2,), dtype=np.float32)

    def _get_indicator_data(self):
        return [self.data[x].values[self.current_step] for x in self.indicators]
         
    def _get_observation(self):
        current_price = self.data['close'].values[self.current_step]
        indicators = self._get_indicator_data()
        balance_fraction = ((self.balance / self.total_balance) - 0.5) * 2 if self.total_balance > 0 else 0

        stakes_present = []
        stakes_data = []
        for i in range(len(self.stakes)):
            stake = self.stakes[i]
            if stake is not None:
                # Calculate stake ROI based on current price
                current_roi = ((current_price - stake['entry_price']) / stake['entry_price']) - (self.buy_fee * stake['entry_price'] * stake['size']) if stake['entry_price'] > 0 else 0
                stakes_data.append(np.float64(current_roi))
                stakes_present.append(1)
            else:
                stakes_data.append(0)  # For empty positions, return 0
                stakes_present.append(0)

        obs = np.array(indicators + [balance_fraction] + stakes_present + stakes_data, dtype=np.float64)  # Use float64 for observation

        # Clamp values to avoid overflow
        obs = np.clip(obs, -1e3, 1e3)
        
        return obs
    
    def _get_reward(self, **kwargs):
        reward = 0
        reward += sum(kwargs['rois']) * 100
        reward += 1 if kwargs['buy_decision'] >= 0.5 else 0
        reward += 1 if any([x >= 0.5 for x in kwargs['sell_actions']]) else 0
        return reward

    def update_data(self, dataframe):
        self.data = dataframe.copy()

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
 
    def step(self, action):

        sell_actions = action[:self.max_stakes]  # Actions for each stake (sell/hold)
        buy_decision = action[self.max_stakes]  # Buy decision (0: no buy, 1: buy)
        
        if buy_decision >= .5:
            buy_fraction = action[self.max_stakes + 1]  # Fraction of balance to buy
        else:
            buy_fraction = 0

        # Get current price and profits
        current_price = self.data['close'].values[self.current_step]
        profits = []
        rois = []

        # Handle selling stakes
        for i in range(len(self.stakes)):
            if sell_actions[i] >= 0.5 and self.stakes[i] is not None:  # Sell this stake if action signals and stake exists
                stake = self.stakes[i]  # Use positional indexing
                
                if stake is not None:  # Ensure valid stake
                    sale_amount = stake['size'] * current_price
                    fee = sale_amount * self.sell_fee
                    
                    # Calculate profit and reward
                    profit = sale_amount - (stake['entry_price'] * stake['size']) - fee
                    roi = profit / (stake['entry_price'] * stake['size']) if stake['entry_price'] > 0 else 0
                    self.balance += sale_amount - fee  # Add sale proceeds to balance

                    # Log trade history
                    self.trade_history.append({'quote_currency': self.quote_currency, 'step': self.current_step, 'type': 'sell', 'price': current_price, 'size': stake['size'], 'profit': profit, 'roi': roi, 'balance': self.balance})

                    # Clear the stake, keeping the placeholder
                    self.stakes[i] = None
                    profits.append(profit)
                    rois.append(roi)

        # Handle buying new stake, minimum buy fraction and minimum balance conditions
        min_buy_fraction = 0.02
        min_stake_size = 0.01
        min_balance = min_stake_size * current_price / (1 - self.buy_fee)  # Minimum balance to buy minimum stake size
        num_empty_positions = sum([1 for stake in self.stakes if self.stakes[stake] is None])

        if buy_decision >= 0.5:
            if buy_fraction > min_buy_fraction and self.balance > min_balance:
                # Look for the first empty stake position (None)
                empty_position = None
                for idx, stake in enumerate(self.stakes):
                    if self.stakes[stake] is None:
                        empty_position = idx
                        break

                if empty_position is not None:  # Only buy if there's an empty position
                    stake_to_buy = self.balance * buy_fraction
                    fee = stake_to_buy * self.buy_fee
                    size = (stake_to_buy - fee) / current_price
                    new_stake = {
                        'base_currency': self.base_currency,
                        'entry_amount': stake_to_buy - fee,
                        'entry_price': current_price,
                        'size': size
                    }
                    self.stakes[empty_position] = new_stake
                    self.balance -= stake_to_buy + fee  # Deduct buy amount and fee from balance
                    self.base_balance += size  # Add to base currency balance
                    self.trade_history.append({'quote_currency': self.quote_currency, 'step': self.current_step, 'type': 'buy', 'price': current_price, 'size': size, 'balance': self.balance})

        # Update total balance
        self.base_currency_balance = self.base_balance * current_price
        self.total_balance = self.quote_balance + self.base_currency_balance

        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        truncated = False

        # Get reward, next observation, and info dictionary
        rew_args = {
            'current_price': current_price,
            'buy_decision': buy_decision,
            'buy_fraction': buy_fraction,
            'num_empty_positions': num_empty_positions,
            'sell_actions': sell_actions,
            'stakes': self.stakes,
            'profits': profits,
            'rois': rois,
            'balance': self.balance,
            'base_currency_balance': self.base_currency_balance,
            'total_balance': self.total_balance,
            'done': done
        }
        reward = self._get_reward(**rew_args)
        obs = self._get_observation()
        info = {'trade_history': self.trade_history}

        return obs, reward, done, truncated, info
    
    def reset(self, *, seed=None, options=None):
        # Seed the environment (for reproducibility)
        super().reset(seed=seed)
        if options is not None:
            if options['step'] is not None:
                self.current_step = options['step']
            else:   
                self.current_step = np.random.randint(0, len(self.data) - 1)
        
        else:
            self.current_step = np.random.randint(0, len(self.data) - 1)

        self.balance = self.initial_balance
        self.base_currency_balance = 0
        self.total_balance = self.initial_balance
        self.stakes = {i: None for i in range(0, self.max_stakes)}
        self.trade_history = []  # Reset trade history
        
        # Return observation and empty info dictionary
        return self._get_observation(), {}
    
    def render(self, mode='print'):
        if self.render_mode is not None:
            
            if self.render_mode == 'print':
                print(f"\tStep {self.current_step} --==-- Current Price: {self.data['close'].values[self.current_step]} === Balance: {self.balance} === Total Balance: {self.total_balance}")
                for i in range(len(self.stakes)):
                    stake = self.stakes[i]
                    if stake is not None:
                        print(f"\t\tStake {i} -- Size: {stake['size']} -- Entry Price: {stake['entry_price']}")
                
            if self.render_mode == 'graph':
                # TODO: Implement graph rendering
                pass
            if self.render_mode == 'file':
                # TODO: Implement file logging for rendering
                pass
        else:
            pass

def load_data(file_path):
    "Read .feather file and return as DataFrame"
    if file_path.endswith('.feather'):
        return pd.read_feather(file_path)
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        raise ValueError("Invalid file format. Please provide a .feather file.")
    
if __name__ == '__main__':
    env = TradingEnv(data_dir, window_size=144, render_mode='human', mode='train')