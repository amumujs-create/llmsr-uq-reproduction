"""Proportional density law PE = mean_PE * r / mean_r (r = AP/T_K), slope fixed by physics.
Compared with fitted linear models, in matched windows for both directions (uses test PE)."""
import sys, numpy as np, openpyxl
rows=list(openpyxl.load_workbook(sys.argv[1],read_only=True,data_only=True)["Sheet1"].iter_rows(values_only=True))
d=np.array([[float(x) for x in r[:5]] for r in rows[1:] if r[0] is not None]); at,v,ap,rh,pe=d.T
r=ap/(at+273.15)
lo,hi=np.quantile(at,[.005,.995]); rng=np.random.default_rng(0)
def lin(cols,m,te):
    X=lambda mm:np.column_stack([np.ones(mm.sum())]+[c[mm] for c in cols])
    b,*_=np.linalg.lstsq(X(m),pe[m],rcond=None); return X(te)@b
print("direction  w   n   d | fit raw  fit density  proportional | implied dPE/dAT: fitted vs proportional")
for w,n,dist in ((2.5,146,10),(2.5,146,15),(5,300,10),(10,1000,10)):
    for name in ("cold->hot","hot->cold"):
        if name=="cold->hot": pool=np.flatnonzero((at>=lo)&(at<=lo+w)); te=(at>=lo+w+dist)&(at<lo+w+dist+2.5)
        else: pool=np.flatnonzero((at>=hi-w)&(at<=hi)); te=(at<=hi-w-dist)&(at>hi-w-dist-2.5)
        acc={"raw":[],"dens":[],"prop":[],"slope_fit":[],"slope_prop":[]}
        for _ in range(20):
            m=np.zeros(len(at),bool); m[rng.choice(pool,min(n,len(pool)),replace=False)]=True
            acc["raw"].append(np.sqrt(np.mean((lin([at,v,ap,rh],m,te)-pe[te])**2)))
            acc["dens"].append(np.sqrt(np.mean((lin([r],m,te)-pe[te])**2)))
            prop=pe[m].mean()*r[te]/r[m].mean()
            acc["prop"].append(np.sqrt(np.mean((prop-pe[te])**2)))
            acc["slope_fit"].append(np.polyfit(at[m],pe[m],1)[0])
            tk=at[m].mean()+273.15; acc["slope_prop"].append(-pe[m].mean()/tk)
        a={k:np.mean(x) for k,x in acc.items()}
        print(f"{name:10s} {w:4.1f} {n:4d} {dist:3.0f} | {a['raw']:7.1f} {a['dens']:11.1f} {a['prop']:13.1f} | {a['slope_fit']:6.2f} vs {a['slope_prop']:6.2f} MW/C")
for nm,mm in (("cold 4-9C",(at>=4)&(at<=9)),("mid 15-20C",(at>=15)&(at<=20)),("hot 29-34C",(at>=29)&(at<=34))):
    print(f"true local dPE/dAT {nm}: {np.polyfit(at[mm],pe[mm],1)[0]:.2f} MW/C ; proportional law predicts {-pe[mm].mean()/(at[mm].mean()+273.15):.2f}")
