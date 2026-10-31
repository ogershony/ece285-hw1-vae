"""
Build report/results.tex and the loss-curve figure from outputs/results.json,
then compile report.tex with tectonic.  Run from the repo root:
    .venv/bin/python report/make_report.py
"""
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT, REP = os.path.join(ROOT, "outputs"), os.path.join(ROOT, "report")
res_path = os.path.join(OUT, "results.json")

if os.path.exists(res_path):
    r = json.load(open(res_path))
    losses = r["training_losses"]; n_train = r["num_train_images"]
    per_img = [l / n_train for l in losses]
    vals = {
        "FID": f"{r['metrics']['FID']:.2f}",
        "IS": f"{r['metrics']['InceptionScore']:.3f}",
        "NumTrain": str(n_train), "NumTest": str(r["num_test_images"]),
        "NumEpochs": str(len(losses)),
        "InitLoss": f"{per_img[0]:.1f}", "FinalLoss": f"{per_img[-1]:.1f}",
        "MinLoss": f"{min(per_img):.1f}", "MinLossEpoch": str(per_img.index(min(per_img)) + 1),
        "Device": r["device"].replace("NVIDIA ", ""),
    }
    # Loss curve: single series -> one hue, no legend, thin line, recessive grid
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.2, 2.0))
    ax.plot(range(1, len(per_img) + 1), per_img, color="#3B6FD4", linewidth=1.6)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss per image (BCE + KL)")
    ax.grid(True, color="#DDDDDD", linewidth=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#999999")
    ax.tick_params(colors="#444444", labelsize=8)
    ax.xaxis.label.set_color("#333333"); ax.yaxis.label.set_color("#333333")
    fig.tight_layout(); fig.savefig(os.path.join(REP, "loss_curve.pdf")); plt.close(fig)
    subprocess.run(["cp", os.path.join(OUT, "generated_grid.png"), os.path.join(REP, "generated_grid.png")], check=True)
    print("results.json found; filled real values")
else:
    vals = dict(FID="--", IS="--", NumTrain="3875", NumTest="390", NumEpochs="100", InitLoss="--",
                FinalLoss="--", MinLoss="--", MinLossEpoch="--", Device="RTX 2080 Ti")
    print("WARNING: outputs/results.json not found; using placeholders", file=sys.stderr)

with open(os.path.join(REP, "results.tex"), "w") as f:
    for k, v in vals.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

subprocess.run(["tectonic", "-X", "compile", "report.tex"], cwd=REP, check=True)
print("built", os.path.join(REP, "report.pdf"))
