import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

def main(proba_csv, out_png):
    #columns: y_true, y_prob
    df = pd.read_csv(proba_csv)
    y = df["y_true"].values
    p = df["y_prob"].values
    thresholds = np.linspace(0.1,0.9,17)
    rows=[]
    for t in thresholds:
        yhat = (p>=t).astype(int)
        tp = ((yhat==1)&(y==1)).sum()
        fp = ((yhat==1)&(y==0)).sum()
        fn = ((yhat==0)&(y==1)).sum()
        prec = tp/(tp+fp) if tp+fp else 0
        rec  = tp/(tp+fn) if tp+fn else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
        rows.append({"threshold":t,"precision":prec,"recall":rec,"f1":f1})
    out = pd.DataFrame(rows)
    out.to_csv(proba_csv.replace(".csv","_thresholds.csv"), index=False)
    out.plot(x="threshold", y=["precision","recall","f1"])
    plt.title("Threshold sweep")
    plt.savefig(out_png, bbox_inches='tight'); plt.close()

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--proba_csv", required=True)
    ap.add_argument("--out_png", required=True)
    args=ap.parse_args()
    main(args.proba_csv, args.out_png)
