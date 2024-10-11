import time
import random
import pandas as pd

from pipeline.api import StockDataAPI
from pipeline.db import PostgreSQL
from utils.logger import logger

CHUNK_SIZE = 10

class DataPoller:
    def __init__(self,
                 macrotrends_api: StockDataAPI,
                 exchanges: list[str] = ["nasdaq", "nyse"],
                 base_ratelimit: int = 10, 
                 ratelimit_buffer: tuple[float] = (0.0, 1.0)) -> None:
        self.exchanges = exchanges
        self.base_ratelimit = base_ratelimit
        self.ratelimit_buffer = ratelimit_buffer  # Adds unpredictability to request timing
        self.api = macrotrends_api
        
    def do_sleep(self) -> None:
        """Sleep for a random time between the base ratelimit and the buffer range."""
        time.sleep(self.base_ratelimit + random.uniform(*self.ratelimit_buffer))
        
    def poll_tickers(self) -> None:
        """Poll all tickers for P/B from the exchanges and store them in the database."""
        
        # Get all tickers from the exchanges
        try:
            all_tickers_df = self.api.get_exchange_tickers(self.exchanges)
        except Exception as e:
            logger.error(f"Error occurred when fetching exchange tickers ({self.exchanges}): {e}")
            return
        
        # Instantiate a new database connection
        with PostgreSQL(db_name='sample_database', 
                        user='sample_user', 
                        password='sample_password', 
                        host='localhost',
                        port='5432') as database:
            
            ratio_history_df_chunk = []
            current_ratio_rows = []
            for i, r in all_tickers_df.iterrows():
                # Get the P/B history
                try:
                    pb_history_df, current_pb = self.api.get_pb_ratio_history(r['symbol'], r['full_name'], r['url'])
                    if pb_history_df is None or current_pb is None:
                        logger.warning(f"No P/B data returned for {r['symbol']}. Skipping...")
                        continue
                    
                    self.do_sleep()
                except Exception as e:
                    logger.warning(f"Could not fetch P/B history for {r['url']}: {e}")
                    continue
                
                # Get the P/E history
                try:
                    pe_history_df, current_pe = self.api.get_pe_ratio_history(r['symbol'], r['full_name'], r['url'])
                    if pe_history_df is None or current_pe is None:
                        logger.warning(f"No P/E data returned for {r['symbol']}. Skipping...")
                        continue
                    
                    self.do_sleep()
                except Exception as e:
                    logger.warning(f"Could not fetch P/E history for {r['url']}: {e}")
                    continue
                    
                # Concatenate the P/B and P/E history
                try:
                    concat_df = pd.concat([pb_history_df.set_index('date'), pe_history_df.set_index('date')], axis=1).reset_index()
                    concat_df.sort_values('date', inplace=True, ascending=False)
                    ratio_history_df_chunk.append(concat_df)
                except Exception as e:
                    logger.warning(f"Could not concatenate P/B and P/E history for {r['symbol']} ({r['full_name']}): {e}")
                    continue
                    
                # Build the row for current P/B and P/E
                current_ratio_rows.append({
                    'symbol': r['symbol'],
                    'last_update': pd.Timestamp.now(),
                    'pb_ratio': current_pb,
                    'pe_ratio': current_pe
                })
                
                # Store the dataframes immediately
                try:
                    if concat_df:
                        database.store_report_dataframes([concat_df])
                except Exception as e:
                    logger.warning(f"An error occurred when storing historical ratio data for {r['symbol']} ({r['full_name']}):", exc_info=True)
                finally:
                    ratio_history_df_chunk.clear()                    
                    
                try:
                    if current_ratio_rows:
                        database.store_current_ratio_dataframes([pd.DataFrame(current_ratio_rows)])
                except Exception as e:
                    logger.warning(f"An error occurred when storing current ratio data for {r['symbol']} ({r['full_name']}):", exc_info=True)
                finally:
                    current_ratio_rows.clear()
    
    def run(self) -> None:
        max_retries = 5  # Maximum number of retries
        base_wait_time = 5  # Base wait time in seconds

        retries = 0
        while True:
            try:
                self.poll_tickers()
                retries = 0  # Reset retries on successful polling
            except KeyboardInterrupt:
                logger.info("Polling stopped by user.")
                break
            except Exception as e:
                logger.error("Error occurred during ticker polling:", exc_info=True)
                retries += 1
                if retries > max_retries:
                    logger.error("Maximum retries reached. Exiting.")
                    break
                wait_time = base_wait_time * (2 ** retries)
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
