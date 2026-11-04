"""
Build the camera-count study dashboard: a static folder you can open in a browser.

    python mpm/build_dashboard.py --scene-dirs out/scene3_collision out/scene7_deformation \
        --results-root $FOURDGS_ROOT/output/mpm_study --out dashboard

Shows, per scene:
  * the camera rig as frustums, coloured by the smallest subset (4/8/12/25/50)
    that contains each camera, plus the held-out test cameras
  * point cloud: MPM particles at t=0, and the trained Gaussian centres per subset
  * GT vs. 4DGS renders from a held-out camera before, at, and after impact
  * PSNR over time per subset, and PSNR / SSIM / LPIPS vs. camera count

Works before training too: missing results just show as "not trained yet".
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

TIER_SUBSETS = [4, 8, 12, 25, 50]


# ------------------------------------------------------------------- data loading
def read_ply_xyz(path, max_points=6000, seed=0):
    with open(path, "rb") as fh:
        header, props, n = [], [], 0
        while True:
            line = fh.readline().decode("ascii").strip()
            header.append(line)
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                _, t, name = line.split()
                props.append((name, {"float": "<f4", "double": "<f8", "uchar": "u1",
                                     "int": "<i4", "uint": "<u4"}[t]))
            elif line == "end_header":
                break
        if "format ascii 1.0" in header:
            raise ValueError("ascii ply not supported")
        data = np.frombuffer(fh.read(n * np.dtype(props).itemsize), dtype=props, count=n)
    xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(float)
    if len(xyz) > max_points:
        xyz = xyz[np.random.default_rng(seed).choice(len(xyz), max_points, replace=False)]
    return xyz


def frustum_lines(c2w, fovy_deg, aspect, depth):
    c2w = np.asarray(c2w)
    h = depth * np.tan(np.radians(fovy_deg) / 2)
    w = h * aspect
    corners_cam = np.array([[-w, -h, -depth], [w, -h, -depth], [w, h, -depth], [-w, h, -depth]])
    apex = c2w[:3, 3]
    corners = (c2w[:3, :3] @ corners_cam.T).T + apex
    seq = [apex, corners[0], corners[1], apex, corners[2], corners[3], apex,
           corners[0], corners[3], None, corners[1], corners[2]]
    xs, ys, zs = [], [], []
    for p in seq + [None]:
        for arr, v in zip((xs, ys, zs), (None, None, None) if p is None else p):
            arr.append(None if v is None else round(float(v), 4))
    return xs, ys, zs


def latest_method(d):
    keys = sorted(d, key=lambda k: int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else -1)
    return keys[-1] if keys else None


def thumb(src, dst, size=320):
    im = Image.open(src).convert("RGB")
    im.thumbnail((size, size))
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, quality=88)
    return dst


def scene_payload(sdir, results_root, out_dir):
    sdir = Path(sdir)
    name = sdir.name
    rig = json.load(open(sdir / "rig.json"))
    sim_meta = json.load(open(sdir / "sim.json"))
    sim = np.load(sdir / "sim.npz")
    scene = sim_meta["scene"]
    n_frames = sim_meta["solver"]["n_frames"]

    # cameras
    train = [c for c in rig["cameras"] if c["split"] == "train"]
    depth = 0.12 * rig["radius"]
    cams = []
    for i, c in enumerate(train):
        tier = next(n for n in TIER_SUBSETS if i < n)
        xs, ys, zs = frustum_lines(c["c2w_gl"], rig["fovy_deg"], rig["width"] / rig["height"], depth)
        cams.append(dict(name=c["name"], tier=tier, x=xs, y=ys, z=zs, pos=c["position"]))
    for c in rig["cameras"]:
        if c["split"] == "test":
            xs, ys, zs = frustum_lines(c["c2w_gl"], rig["fovy_deg"], rig["width"] / rig["height"], depth)
            cams.append(dict(name=c["name"], tier="test", x=xs, y=ys, z=zs, pos=c["position"]))

    # point clouds
    pobj = sim["particle_obj"]
    p0 = sim["particles"][0]
    colors = ["rgb(%d,%d,%d)" % tuple(int(255 * v) for v in scene["objects"][k]["color"]) for k in pobj]
    clouds = {"sim": dict(x=p0[:, 0].round(4).tolist(), y=p0[:, 1].round(4).tolist(),
                          z=p0[:, 2].round(4).tolist(), color=colors)}

    # frames to show: before impact, peak strain, after
    strain = sim["max_strain"]
    peak = int(np.argmax(strain))
    first = int(np.argmax(strain > 0.02)) if np.any(strain > 0.02) else peak
    test_cams = [c["name"] for c in rig["cameras"] if c["split"] == "test"]
    show_cam = test_cams[0]
    # snap to frames that exist in the exported test set
    avail = list(range(n_frames))
    idx0 = sdir / "4dgs" / f"cams_{rig['subsets'][0]:02d}" / "test_index.json"
    if idx0.exists():
        avail = sorted({e["frame"] for e in json.load(open(idx0))["test"]}) or avail
    snap = lambda f: min(avail, key=lambda a: abs(a - f))
    peak = snap(peak)
    show_frames = sorted({snap(max(0, first - 12)), peak, snap(min(n_frames - 1, peak + 20))})
    img_dir = out_dir / "img" / name
    gt = [str(thumb(sdir / "images" / show_cam / f"{f:05d}.png", img_dir / f"gt_{f:05d}.jpg").relative_to(out_dir))
          if (sdir / "images" / show_cam / f"{f:05d}.png").exists() else None for f in show_frames]

    metrics, renders, per_time = {}, {}, {}
    for n in rig["subsets"]:
        exp = Path(results_root) / name / f"cams_{n:02d}" if results_root else None
        idx_file = sdir / "4dgs" / f"cams_{n:02d}" / "test_index.json"
        if exp and (exp / "point_cloud").exists():
            its = sorted((exp / "point_cloud").glob("iteration_*"), key=lambda p: int(p.name.split("_")[-1]))
            if its and (its[-1] / "point_cloud.ply").exists():
                xyz = read_ply_xyz(its[-1] / "point_cloud.ply")
                clouds[f"gaussians_{n}"] = dict(x=xyz[:, 0].round(4).tolist(), y=xyz[:, 1].round(4).tolist(),
                                                z=xyz[:, 2].round(4).tolist(), color=None)
        if not (exp and (exp / "results.json").exists()):
            continue
        res = json.load(open(exp / "results.json"))
        m = latest_method(res)
        metrics[n] = {k: res[m][k] for k in ("PSNR", "SSIM", "LPIPS-vgg") if k in res[m]}
        if idx_file.exists() and (exp / "per_view.json").exists():
            pv = json.load(open(exp / "per_view.json"))[m]["PSNR"]
            index = json.load(open(idx_file))["test"]
            acc = {}
            for e in index:
                key = f"{e['index']:05d}.png"
                if key in pv:
                    acc.setdefault(e["frame"], []).append(pv[key])
            per_time[n] = [[f, float(np.mean(v))] for f, v in sorted(acc.items())]
            lookup = {(e["camera"], e["frame"]): e["index"] for e in index}
            row = []
            for f in show_frames:
                i = lookup.get((show_cam, f))
                src = exp / "test" / m / "renders" / f"{i:05d}.png" if i is not None else None
                row.append(str(thumb(src, img_dir / f"cams{n:02d}_{f:05d}.jpg").relative_to(out_dir))
                           if src is not None and src.exists() else None)
            renders[n] = row

    return dict(
        name=name, subsets=rig["subsets"], fps=scene["fps"], n_frames=n_frames,
        target=rig["target"], cams=cams, clouds=clouds,
        objects=[dict(name=o["name"], E=o["E"], nu=o["nu"], color=o["color"]) for o in scene["objects"]],
        static=[s["name"] for s in scene["static_objects"]],
        show_cam=show_cam, show_frames=show_frames, peak_frame=peak, gt=gt,
        renders=renders, metrics=metrics, per_time=per_time,
        strain=[round(float(s), 4) for s in strain],
        solver=sim_meta["solver"],
    )


# --------------------------------------------------------------------------- html
HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Camera count study</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"></script>
<style>
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--test:#eb6834;
--t4:#86b6ef;--t8:#5598e7;--t12:#2a78d6;--t25:#1c5cab;--t50:#104281;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--test:#d95926;--t4:#184f95;--t8:#256abf;--t12:#3987e5;--t25:#6da7ec;--t50:#b7d3f6;}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;
--test:#d95926;--t4:#184f95;--t8:#256abf;--t12:#3987e5;--t25:#6da7ec;--t50:#b7d3f6;}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1200px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:0 0 8px}
.sub{color:var(--ink2);margin:0 0 20px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:16px;margin-bottom:16px}
.row{display:grid;gap:16px;grid-template-columns:repeat(3,minmax(0,1fr))}
.row2{display:grid;gap:16px;grid-template-columns:minmax(0,3fr) minmax(0,2fr)}
@media (max-width:860px){.row,.row2{grid-template-columns:1fr}}
.tabs{display:flex;gap:8px;margin:8px 0 16px;flex-wrap:wrap}
.tabs button,.seg button{font:inherit;border:1px solid var(--ring);background:var(--surface);color:var(--ink);
border-radius:8px;padding:6px 12px;cursor:pointer}
.tabs button[aria-selected="true"],.seg button[aria-pressed="true"]{background:var(--ink);color:var(--surface)}
.seg{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px}
.plot{width:100%;height:300px}.plot3d{width:100%;height:460px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:6px 8px;border-bottom:1px solid var(--grid)}th:first-child,td:first-child{text-align:left}
th{color:var(--ink2);font-weight:600}
.grid{display:grid;gap:8px;grid-template-columns:72px repeat(var(--cols),minmax(0,1fr));align-items:center}
.grid img{width:100%;border-radius:6px;border:1px solid var(--ring);display:block}
.ph{aspect-ratio:1;border-radius:6px;border:1px dashed var(--axis);display:flex;align-items:center;
justify-content:center;color:var(--muted);font-size:12px;text-align:center;padding:4px}
.lab{color:var(--ink2);font-size:12px}.muted{color:var(--muted)}
.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--ink2);font-size:12px;margin-top:6px}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
</style></head><body><main>
<h1>Camera count study</h1>
<p class="sub">How many cameras are needed? 4DGS trained on nested subsets of 4, 8, 12, 25 and 50 cameras,
scored on the same 8 held-out cameras. Scenes simulated with MPM, rendered in Blender.</p>

<section class="card"><h2>Image quality vs. number of training cameras</h2>
<div class="row"><div id="m-PSNR" class="plot"></div><div id="m-SSIM" class="plot"></div><div id="m-LPIPS-vgg" class="plot"></div></div>
<div id="mtable"></div></section>

<div class="tabs" role="tablist" id="tabs"></div>
<div id="scene"></div>
</main>
<script>
const DATA = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const tierVar = t => t === "test" ? "--test" : "--t" + t;
const font = () => ({family:'system-ui,-apple-system,"Segoe UI",sans-serif', size:12, color:css("--ink2")});
function layout(extra){return Object.assign({paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,0,0,0)",font:font(),
  margin:{l:52,r:16,t:30,b:44},hovermode:"closest",showlegend:true,legend:{orientation:"h",y:-0.25},
  xaxis:{gridcolor:css("--grid"),linecolor:css("--axis"),zeroline:false},
  yaxis:{gridcolor:css("--grid"),linecolor:css("--axis"),zeroline:false}},extra||{});}
const cfg = {displayModeBar:false,responsive:true};
const scenes = DATA.scenes;

function summary(){
  const metrics = [["PSNR","PSNR (dB), higher is better"],["SSIM","SSIM, higher is better"],["LPIPS-vgg","LPIPS, lower is better"]];
  const sc = ["--s1","--s2"];
  for (const [k,title] of metrics){
    const traces = scenes.map((s,i)=>{const ns=s.subsets.filter(n=>s.metrics[n]&&s.metrics[n][k]!=null);
      return {x:ns,y:ns.map(n=>s.metrics[n][k]),name:s.name,mode:"lines+markers",
      line:{width:2,color:css(sc[i%2])},marker:{size:8,color:css(sc[i%2]),line:{width:2,color:css("--surface")}},
      hovertemplate:"%{x} cameras<br>"+k+" %{y:.3f}<extra>"+s.name+"</extra>"};});
    const any = traces.some(t=>t.x.length);
    Plotly.react("m-"+k, traces, layout({title:{text:title,font:{size:13,color:css("--ink")}},
      xaxis:{type:"log",tickvals:[4,8,12,25,50],title:{text:"training cameras"},gridcolor:css("--grid"),linecolor:css("--axis")},
      yaxis:{gridcolor:css("--grid"),linecolor:css("--axis"),zeroline:false},
      annotations:any?[]:[{text:"not trained yet",showarrow:false,xref:"paper",yref:"paper",x:.5,y:.5,font:{color:css("--muted")}}]}),cfg);
  }
  let h = "<table><thead><tr><th>scene</th><th>cameras</th><th>PSNR</th><th>SSIM</th><th>LPIPS</th></tr></thead><tbody>";
  for (const s of scenes) for (const n of s.subsets){const m=s.metrics[n]||{};const f=(v,d)=>v==null?'<span class="muted">-</span>':v.toFixed(d);
    h += `<tr><td>${s.name}</td><td>${n}</td><td>${f(m.PSNR,2)}</td><td>${f(m.SSIM,4)}</td><td>${f(m["LPIPS-vgg"],4)}</td></tr>`;}
  document.getElementById("mtable").innerHTML = h + "</tbody></table>";
}

function sceneView(si){
  const s = scenes[si];
  document.querySelectorAll("#tabs button").forEach((b,i)=>b.setAttribute("aria-selected",i===si));
  const fmtE = e => e>=1e6?(e/1e6).toFixed(1)+" MPa":(e/1e3).toFixed(0)+" kPa";
  const mats = s.objects.map(o=>`<span class="lab"><span class="sw" style="background:rgb(${o.color.map(c=>c*255).join(",")})"></span>${o.name}: E = ${fmtE(o.E)}, ν = ${o.nu}</span>`).join(" &nbsp; ")
    + (s.static.length?` &nbsp; <span class="lab">${s.static.join(", ")}: rigid</span>`:"");
  const cloudKeys = Object.keys(s.clouds);
  document.getElementById("scene").innerHTML = `
  <section class="card"><h2>${s.name}</h2><div>${mats}</div>
  <p class="lab">MPM: grid ${s.solver.grid.join("×")}, dx ${s.solver.dx} m, ${s.solver.particles.toLocaleString()} particles, ${s.n_frames} frames at ${s.fps} fps. Peak strain at frame ${s.peak_frame}.</p></section>
  <div class="row2">
   <section class="card"><h2>Camera rig and point cloud</h2>
    <div class="seg" id="cloudsel">${cloudKeys.map((k,i)=>`<button aria-pressed="${i===0}" data-k="${k}">${k==="sim"?"MPM particles (t=0)":"Gaussians, "+k.split("_")[1]+" cams"}</button>`).join("")}</div>
    <div id="rig" class="plot3d"></div>
    <div class="legend">${[4,8,12,25,50].filter(n=>n<=Math.max(...s.subsets)).map(n=>`<span><span class="sw" style="background:${css("--t"+n)}"></span>in ${n}-cam set</span>`).join("")}<span><span class="sw" style="background:${css("--test")}"></span>held-out test</span></div>
   </section>
   <section class="card"><h2>PSNR over time (held-out cameras)</h2><div id="ptime" class="plot" style="height:460px"></div></section>
  </div>
  <section class="card"><h2>Renders from ${s.show_cam} (held out)</h2><div id="renders"></div></section>`;

  let cloud = cloudKeys[0];
  const drawRig = () => {
    const traces = [];
    for (const t of [4,8,12,25,50,"test"]){
      const cs = s.cams.filter(c=>c.tier===t); if(!cs.length) continue;
      const x=[],y=[],z=[],txt=[];
      for (const c of cs){x.push(...c.x);y.push(...c.y);z.push(...c.z);txt.push(...c.x.map(()=>c.name));}
      traces.push({type:"scatter3d",mode:"lines",x,y,z,text:txt,name:t==="test"?"test":"≤"+t,
        line:{width:3,color:css(tierVar(t))},hovertemplate:"%{text}<extra></extra>",showlegend:false});
    }
    const c = s.clouds[cloud];
    traces.push({type:"scatter3d",mode:"markers",x:c.x,y:c.y,z:c.z,name:"points",showlegend:false,
      marker:{size:1.6,color:c.color||css("--ink2"),opacity:.8},hoverinfo:"skip"});
    Plotly.react("rig",traces,{paper_bgcolor:"rgba(0,0,0,0)",font:font(),margin:{l:0,r:0,t:0,b:0},
      scene:{aspectmode:"data",xaxis:{visible:false},yaxis:{visible:false},zaxis:{visible:false},
      camera:{eye:{x:1.3,y:1.3,z:.9}}}},cfg);
  };
  drawRig();
  document.querySelectorAll("#cloudsel button").forEach(b=>b.onclick=()=>{cloud=b.dataset.k;
    document.querySelectorAll("#cloudsel button").forEach(x=>x.setAttribute("aria-pressed",x===b));drawRig();});

  const pt = s.subsets.filter(n=>s.per_time[n]).map(n=>({x:s.per_time[n].map(p=>p[0]/s.fps),y:s.per_time[n].map(p=>p[1]),
    name:n+" cams",mode:"lines",line:{width:2,color:css("--t"+n)},hovertemplate:"t=%{x:.2f}s<br>%{y:.2f} dB<extra>"+n+" cams</extra>"}));
  Plotly.react("ptime",pt,layout({hovermode:"x unified",
    xaxis:{title:{text:"time (s)"},gridcolor:css("--grid"),linecolor:css("--axis")},
    yaxis:{title:{text:"PSNR (dB)"},gridcolor:css("--grid"),linecolor:css("--axis")},
    shapes:[{type:"line",x0:s.peak_frame/s.fps,x1:s.peak_frame/s.fps,yref:"paper",y0:0,y1:1,line:{color:css("--axis"),dash:"dot",width:1}}],
    annotations:[{x:s.peak_frame/s.fps,yref:"paper",y:1,text:"peak strain",showarrow:false,yanchor:"bottom",font:{size:11,color:css("--muted")}}]
      .concat(pt.length?[]:[{text:"not trained yet",showarrow:false,xref:"paper",yref:"paper",x:.5,y:.5,font:{color:css("--muted")}}])}),cfg);

  const cell = (src,alt) => src?`<img src="${src}" alt="${alt}" loading="lazy">`:`<div class="ph">not trained yet</div>`;
  const cols = ["ground truth"].concat(s.subsets.map(n=>n+" cams"));
  let g = `<div class="grid" style="--cols:${cols.length}"><div></div>` + cols.map(c=>`<div class="lab">${c}</div>`).join("");
  s.show_frames.forEach((f,i)=>{
    g += `<div class="lab">${(f/s.fps).toFixed(2)} s${f===s.peak_frame?"<br>impact":""}</div>` + cell(s.gt[i],"ground truth")
      + s.subsets.map(n=>cell((s.renders[n]||[])[i],n+" cameras")).join("");
  });
  document.getElementById("renders").innerHTML = g + "</div>";
}

document.getElementById("tabs").innerHTML = scenes.map((s,i)=>`<button role="tab" data-i="${i}">${s.name}</button>`).join("");
document.querySelectorAll("#tabs button").forEach(b=>b.onclick=()=>sceneView(+b.dataset.i));
let cur = 0;
document.querySelectorAll("#tabs button").forEach(b=>b.addEventListener("click",()=>cur=+b.dataset.i));
const redraw = () => {summary(); sceneView(cur);};
redraw();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
new MutationObserver(redraw).observe(document.documentElement,{attributes:true,attributeFilter:["data-theme"]});
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-dirs", nargs="+", required=True)
    ap.add_argument("--results-root", default=None,
                    help="4DGaussians output folder of the study, e.g. $FOURDGS_ROOT/output/mpm_study")
    ap.add_argument("--out", default="dashboard")
    args = ap.parse_args()
    out = Path(args.out)
    if (out / "img").exists():
        shutil.rmtree(out / "img")
    out.mkdir(parents=True, exist_ok=True)
    data = dict(scenes=[scene_payload(s, args.results_root, out) for s in args.scene_dirs])
    (out / "index.html").write_text(HTML.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    (out / "data.json").write_text(json.dumps(data))
    print(f"wrote {out / 'index.html'}")


if __name__ == "__main__":
    main()
