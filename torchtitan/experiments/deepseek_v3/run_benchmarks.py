import subprocess
import re
import csv

# 1. Sequence lengths to test
seq_lens = [32, 64, 128, 256]
batch_sizes = [4, 8, 16, 32]
# 2. Storage for parsed results
results = []

for seq_len in seq_lens:
    for batch_size in batch_sizes:
        cmd = [
            "torchrun", "--standalone", "--nproc-per-node=4",
            "benchmark_moe_overlap.py",
            "--num_layers=2",
            f"--batch_size={batch_size}",
            f"--seq_len={seq_len}",
        "--iterations=2"
    ]
    print(f"\n▶ Running benchmark for seq_len={seq_len}...")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout

    # 3. Extract the numbers with regex
    van_match = re.search(r"Vanilla \(no overlap\):\s*([\d\.]+)\s*ms/iter", out)
    ovl_match = re.search(r"Comet \(with overlap\):\s*([\d\.]+)\s*ms/iter", out)
    tok_van = re.search(r"Tokens/sec vanilla:\s*([\d\.]+)", out)
    tok_ovl = re.search(r"Tokens/sec Comet:\s*([\d\.]+)", out)

    if not (van_match and ovl_match and tok_van and tok_ovl):
        print(f"⚠️  Failed to parse output for seq_len={seq_len}")
        print(out)
        continue

    vanilla_time = float(van_match.group(1))
    overlap_time = float(ovl_match.group(1))
    tokens_vanilla = float(tok_van.group(1))
    tokens_comet  = float(tok_ovl.group(1))

    # 4. Compute speedup & improvement
    speedup     = vanilla_time / overlap_time
    improvement = (vanilla_time - overlap_time) / vanilla_time * 100.0

    results.append({
        'seq_len':               seq_len,
        'batch_size':            batch_size,
        'vanilla_time_ms':       vanilla_time,
        'overlap_time_ms':       overlap_time,
        'speedup':               speedup,
        'improvement_pct':       improvement,
        'tokens_sec_vanilla':    tokens_vanilla,
        'tokens_sec_comet':      tokens_comet,
    })

# 5. Write out CSV
csv_path = "benchmark_results_batch.csv"
with open(csv_path, "w", newline="") as f:
    fieldnames = [
        'seq_len',
        'batch_size',
        'vanilla_time_ms',
        'overlap_time_ms',
        'speedup',
        'improvement_pct',
        'tokens_sec_vanilla',
        'tokens_sec_comet',
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"\n✅ Done! Results written to {csv_path}")