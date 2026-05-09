from smartsignal.workflow.pipeline import SmartSignalPipeline

tickers = [
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","JPM","V","UNH",
    "JNJ","XOM","WMT","MA","PG","HD","CVX","MRK","ABBV","PEP",
]

pipe = SmartSignalPipeline(
    start_date="2018-01-01",
    n_long=5,
    n_short=5,
    train_years=2,
)
result = pipe.run(tickers=tickers)
result.print_summary()
result.plot(save_dir="./charts/try")