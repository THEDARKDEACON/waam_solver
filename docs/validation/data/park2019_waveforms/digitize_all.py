"""Digitise Park 2019 welding-signal figures (3b flat, 4b overhead, 8b vertical down)."""
import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import medfilt
FIGS = {
 "flat":          dict(png="p-05.png", y0=1274.5, y1=843.0, x0=323, t0=2.970, x1=1314, t1=3.030, dk=110,
                       tv=(2.9752, 3.0270),
                       masks=[(835, 985, 852, 885), (1000, 1170, 852, 902), (1190, 1345, 850, 905),
                              (630, 805, 1240, 1270), (670, 790, 1222, 1234), (940, 1085, 1238, 1268), (890, 948, 1224, 1250)]),
 "overhead":      dict(png="p-06.png", y0=1289.0, y1=855.5, x0=319, t0=3.750, x1=1346, t1=3.805, dk=110,
                       tv=(3.7525, 3.8030),
                       masks=[(1205, 1345, 862, 908), (828, 962, 1146, 1166), (955, 1255, 1252, 1288)]),
 "vertical_down": dict(png="p-08.png", y0=1876.5, y1=1446.0, x0=353, t0=3.120, x1=1290, t1=3.175, dk=170,
                       tv=(3.1205, 3.1735),
                       masks=[(1190, 1335, 1455, 1505), (1055, 1300, 1776, 1814)]),
}
out = {}
for name, f in FIGS.items():
    im = np.asarray(Image.open(f["png"]).convert("RGB")).astype(int)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    th = f["dk"]
    dark = (r < th) & (g < th) & (b < th) & (np.abs(r - g) < 40) & (np.abs(g - b) < 40)
    red = (r > 170) & (r - g > 70) & (r - b > 70)
    mask = np.zeros_like(dark)
    for xa, xb, ya, yb in f["masks"]:
        mask[ya:yb, xa:xb] = True
    ytop, ybot = int(f["y1"]) + 3, int(f["y0"]) - 2
    xs = np.arange(f["x0"] + 3, f["x1"] - 3)
    I = np.full(len(xs), np.nan); U = np.full(len(xs), np.nan)
    for n, x in enumerate(xs):
        ys = np.where(dark[ytop:ybot, x] & ~mask[ytop:ybot, x])[0] + ytop
        if len(ys): I[n] = (f["y0"] - ys.mean()) / (f["y0"] - f["y1"]) * 400
        yr = np.where(red[ytop:ybot, x] & ~mask[ytop:ybot, x])[0] + ytop
        if len(yr): U[n] = (f["y0"] - yr.mean()) / (f["y0"] - f["y1"]) * 50
    t = f["t0"] + (xs - f["x0"]) / (f["x1"] - f["x0"]) * (f["t1"] - f["t0"])
    ok = ~np.isnan(I); I = medfilt(np.interp(t, t[ok], I[ok]), 5)
    ok = ~np.isnan(U); U = np.interp(t, t[ok], U[ok])
    v = (t > f["tv"][0]) & (t < f["tv"][1])
    rise = np.where((I[1:] > 200) & (I[:-1] <= 200) & v[1:])[0]
    a, z = rise[0], rise[-1]
    Iw, Uw = I[a:z], U[a:z]
    per = [(I[p:q] * U[p:q]).mean() for p, q in zip(rise[:-1], rise[1:])]
    res = dict(periods=len(rise) - 1, freq_hz=(len(rise) - 1) / (t[z] - t[a]),
               I_mean=Iw.mean(), I_rms=np.sqrt((Iw ** 2).mean()), U_mean=Uw.mean(),
               P_mean=(Iw * Uw).mean(), P_sd_period=float(np.std(per)))
    out[name] = res
    print(f"{name:14s} periods {res['periods']}  f {res['freq_hz']:.1f} Hz  I_mean {res['I_mean']:.1f} A  "
          f"I_rms {res['I_rms']:.1f} A  U_mean {res['U_mean']:.2f} V  mean(UI) {res['P_mean']:.0f} W (period sd {res['P_sd_period']:.0f})")
    np.savetxt(f"park_{name}_waveform.csv", np.c_[t, I, U], delimiter=",", header="t_s,I_A,U_V", comments="")
    crop = (f["x0"] - 25, int(f["y1"]) - 15, f["x1"] + 25, int(f["y0"]) + 15)
    ov = Image.open(f["png"]).convert("RGB").crop(crop); d = ImageDraw.Draw(ov)
    Y = lambda val, full: f["y0"] - val / full * (f["y0"] - f["y1"]) - crop[1]
    for n in range(len(xs) - 1):
        d.line([(xs[n] - crop[0], Y(I[n], 400)), (xs[n + 1] - crop[0], Y(I[n + 1], 400))], fill=(0, 170, 255))
        d.line([(xs[n] - crop[0], Y(U[n], 50)), (xs[n + 1] - crop[0], Y(U[n + 1], 50))], fill=(0, 200, 0))
    ov.save(f"overlay_{name}.png")
import json; json.dump(out, open("park_waveforms.json", "w"), indent=1)
