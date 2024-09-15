import streamlit as st


# Main function
def main():
    st.set_page_config(layout="wide")
    
    st.markdown("""
    ## Welcome to Henry's Screening Tools!

    This tool is designed to help you screen stocks based on specific criteria using Macrotrends data. Here's a brief overview of what you can do.

    ### Key Features:
    
    - **Stock Screener**: Filter stocks based on criteria like price-to-book (P/B) and price-to-earnings (P/E) ratios, years of positive P/B & P/E history, and more.
    - **Price & Time Checker**: Analyze stocks based on debt levels over different holding periods.

    ### How to Use:
    
    👈 Please select a page from the sidebar to get started. You are currently on the **Home** page, but you can select from the **Screener** and **Price & Time Checker** pages.

    ### Questions & Concerns:
    
    If you are experiencing any issues or have questions about the tool, please reach out to the developer:
    
    - **Phone**: 972-571-3364
    - **Email**: hschickdevs@gmail.com
    """)

if __name__ == "__main__":
    main()