"""Regenerate perfield.pdf for the IE 643 report.  python make_figures.py
Numbers are the final model's per-field VRDU recall (runs\\visual\\analysis\\failure_report.md, section E)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np

rows = [("gross_amount",0.959,0.738),("registrant_name",0.942,0.537),("registration_num",0.883,0.873),
        ("sub_amount",0.867,0.462),("contract_num",0.822,0.818),("program_end_date",0.796,0.627),
        ("channel",0.787,0.250),("file_date",0.764,0.755),("program_start_date",0.756,0.494),
        ("advertiser",0.748,0.703),("agency",0.738,0.637),("program_desc",0.677,0.325),
        ("product",0.609,0.550),("property",0.585,0.258),("foreign_principle_name",0.568,0.443),
        ("flight_from",0.532,0.373),("flight_to",0.521,0.394),("signer_name",0.351,0.255),
        ("tv_address",0.251,None),("signer_title",0.084,0.062)]
names=[r[0] for r in rows]; vr=[r[1] for r in rows]
kr=[0.0 if r[2] is None else r[2] for r in rows]; mask=[r[2] is None for r in rows]
y=np.arange(len(rows)); h=0.38
fig,ax=plt.subplots(figsize=(9.2,6.6))
ax.barh(y-h/2,vr,height=h,color="#2f6fb5",label="value recall  (text found on the page)",zorder=3)
ax.barh(y+h/2,kr,height=h,color="#d2683a",label="key-matched  (found and correctly named)",zorder=3)
for i,(v,k,m) in enumerate(zip(vr,kr,mask)):
    ax.text(1.02,i,f"{round(v*100):d} / "+("–" if m else f"{round(k*100):d}"),
            va="center",ha="left",fontsize=8.5,color="#444")
    if m: ax.text(0.012,i+h/2,"no alias mapping",va="center",ha="left",fontsize=7.5,
                  color="#777",style="italic",zorder=4)
ax.set_yticks(y); ax.set_yticklabels(names,fontsize=9); ax.invert_yaxis()
ax.set_xlim(0,1.0); ax.set_xticks([0,.25,.5,.75,1.0])
ax.set_xticklabels(["0%","25%","50%","75%","100%"],fontsize=9)
ax.xaxis.grid(True,color="#d8d8d8",lw=0.7,zorder=0); ax.set_axisbelow(True)
for s in ("top","right","left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#999"); ax.tick_params(axis="y",length=0)
ax.legend(loc="lower right",frameon=False,fontsize=9)
ax.set_xlabel("recall on unseen-template VRDU documents (final model, image stream on)",fontsize=9,color="#444")
fig.subplots_adjust(left=0.22,right=0.90,top=0.98,bottom=0.08)
fig.savefig("perfield.pdf"); print("perfield.pdf written")
