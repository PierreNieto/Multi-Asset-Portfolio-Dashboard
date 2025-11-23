    bench_df = yf.download(
        benchmark,
        start=port_norm.index[0],
        end=port_norm.index[-1]
    )

    benchmark_prices = bench_df.get("Close", bench_df.get("Adj Close"))
    benchmark_prices = benchmark_prices.reindex(port_norm.index).ffill()

    bench_