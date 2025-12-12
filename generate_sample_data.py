import argparse
import numpy as np
import pandas as pd

def make_row():
    packet_size = np.random.randint(40, 2000)
    failed_logins = np.random.poisson(1)
    request_frequency = np.random.poisson(20)
    src_bytes = np.random.randint(0, 200000)
    dst_bytes = np.random.randint(0, 200000)
    duration = np.random.exponential(1.0)

    # -----------------------
    # REALISTIC ATTACK RULES
    # -----------------------
    attack = 0

    # Brute-force login attack
    if failed_logins > 7:
        attack = 1

    # DDoS-like pattern: massive packets + extreme frequency
    if packet_size > 1500 and request_frequency > 150:
        attack = 1

    # Data exfiltration pattern
    if src_bytes > 150000 and dst_bytes > 150000:
        attack = 1

    # Very short duration bursts indicate scanning
    if duration < 0.1 and request_frequency > 100:
        attack = 1

    return [
        packet_size, failed_logins, request_frequency,
        src_bytes, dst_bytes, duration, attack
    ]



def main(out, n):
    cols = ['packet_size','failed_logins','request_frequency','src_bytes','dst_bytes','duration','Attack_Label']
    rows = [make_row() for _ in range(n)]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='sample_cyber_data.csv')
    parser.add_argument('--n', type=int, default=1000)
    args = parser.parse_args()
    main(args.out, args.n)