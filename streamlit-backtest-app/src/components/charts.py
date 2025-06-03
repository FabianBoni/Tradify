from matplotlib import pyplot as plt

def plot_equity_curve(equity_curve):
    plt.figure(figsize=(10, 5))
    plt.plot(equity_curve, label='Equity Curve')
    plt.title('Strategy Equity Curve')
    plt.xlabel('Time')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid()
    plt.show()

def plot_performance_metrics(metrics):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].bar(metrics.keys(), metrics.values())
    ax[0].set_title('Performance Metrics')
    ax[0].set_ylabel('Value')
    ax[0].grid()

    ax[1].pie(metrics.values(), labels=metrics.keys(), autopct='%1.1f%%')
    ax[1].set_title('Performance Metrics Distribution')

    plt.tight_layout()
    plt.show()