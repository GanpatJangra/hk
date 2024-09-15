# Henry's Screening Tools

> **DASHBOARD URL:** [http://172.105.101.61:8501/](http://172.105.101.61:8501/)

> **DASHBOARD PASSWORD:** `5Sundry8!!`

This program is designed to help Henry with his stock screening and analysis. It includes a web interface built with Streamlit and a PostgreSQL database to store the fundamental stock data.

It includes two pages:

1. The main page is a stock screener that filters stocks based on the price-to-book (P/B) ratio and other criteria.

2. The secondary page allows the upload of a data file with stock data to check for high and low debt stocks that have reached the target price and/or have been held for a certain period.

## Screener Criteria:

For all stocks in the `NYSE` and `NASDAQ`, they must meet the following criteria to be included in the screen:

* Not a preferred stock, right, warrant, or ticker symbol with 5 letters

* \>= 7 years of positive P/B ratio history

* Current P/B ratio is less than 2

* Current price-to-book (P/B) ratio is less than the lower of either the 3-year average P/B ratio or the 7+ year average P/B ratio, multiplied by the margin of safety factor.

### Output Values

For each stock that matches the screening criteria:

* _see [example2.1.rtf](example2.1.rtf)_

* Need a button to copy these results to clipboard to be pasted where he pleases 

## Installation & Setup Instructions

### 1. PostgreSQL Database Setup

#### Install and start the database:

1. Install Docker on your machine.

2. Pull the PostgreSQL Docker image:

    ```bash
    docker pull postgres
    ```

3. Run the PostgreSQL container:

    ```bash
    docker run --name sample_postgres_db -e POSTGRES_PASSWORD=sample_password -e POSTGRES_USER=sample_user -e POSTGRES_DB=sample_database -p 5432:5432 -d postgres
    ```

4. Open the port `5432` on your machine to allow connections to the PostgreSQL database.

5. Enter the PostgreSQL shell

    ```bash
    docker exec -it sample_postgres_db psql -U sample_user -d sample_database
    ```

    Query to get # rows and cols:

    ```
    WITH row_count AS (
        SELECT COUNT(*) AS num_rows 
        FROM macrotrends_pe_pb_hist
    ), col_count AS (
        SELECT COUNT(*) AS num_cols 
        FROM information_schema.columns 
        WHERE table_name = 'macrotrends_pe_pb_hist'
    )
    SELECT 
        row_count.num_rows, 
        col_count.num_cols 
    FROM 
        row_count, col_count;
    ```

### 2. Run the Data Polling Script

1. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

2. Run the data polling script:

    ```bash
    python3 src/Poller.py
    ```

This process will take quite a few hours to complete, but the Streamlit dashboard is still able to run while it is processing.

### 3. Run the Streamlit Dashboard

1. CD into the project directory:

    ```bash
    cd src/
    ```

2. Run the Streamlit dashboard:

    ```bash
    streamlit run Home.py
    ```