"""Reverse CCPP split: train 30<=AT<=35, request 5<=AT<=20. Request PE is never read."""
import sys
import numpy as np, openpyxl
rows=list(openpyxl.load_workbook(sys.argv[1],read_only=True,data_only=True)["Sheet1"].iter_rows(values_only=True))
d=np.array([[float(v) for v in r[:5]] for r in rows[1:] if r[0] is not None]); at,v,ap,rh,pe=d.T
tr=(at>=30)&(at<=35); te=(at>=5)&(at<=20)
def X(k,m):
    t=at[m]+273.15
    c={"linear":[at[m],v[m],ap[m],rh[m]],"inv_t":[1/t,v[m],ap[m],rh[m]],"quad":[at[m],at[m]**2,v[m],ap[m],rh[m]]}[k]
    return np.column_stack([np.ones(m.sum())]+c)
print(f"train rows={tr.sum()} (AT {at[tr].min():.1f}~{at[tr].max():.1f}), request rows={te.sum()} (AT {at[te].min():.1f}~{at[te].max():.1f})")
fit={k:np.linalg.lstsq(X(k,tr),pe[tr],rcond=None)[0] for k in ("linear","inv_t","quad")}
sd=np.std(pe[tr]-X("linear",tr)@fit["linear"],ddof=5)
base=X("linear",te)@fit["linear"]
print(f"train residual SD={sd:.2f} MW, train PE range={np.ptp(pe[tr]):.1f} MW")
for k in ("inv_t","quad"):
    dd=X(k,te)@fit[k]-base
    print(f"{k}: RMS diff vs linear={np.sqrt(np.mean(dd**2)):.2f} MW, mean={dd.mean():+.2f}, by AT bin:",
          [round(float(dd[(at[te]>=lo)&(at[te]<lo+5)].mean()),2) for lo in (5,10,15)])
