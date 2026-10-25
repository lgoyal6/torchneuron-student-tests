"""Stream every layer of the official torch-neuronx DLC and list Python packages, without extracting to disk."""
import json, re, sys, tarfile, urllib.request

REPO, TAG = "neuron/torch-neuronx", "2.14.0-neuronx-py314-sdk2.33.0-ubuntu26.04"
tok = json.load(urllib.request.urlopen(f"https://public.ecr.aws/token/?scope=repository:{REPO}:pull"))["token"]
H = {"Authorization": f"Bearer {tok}",
     "Accept": "application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json"}
man = json.load(urllib.request.urlopen(urllib.request.Request(f"https://public.ecr.aws/v2/{REPO}/manifests/{TAG}", headers=H)))
pat = re.compile(r"site-packages/([A-Za-z0-9_.+-]+)\.dist-info/METADATA$")
found, files = {}, set()
for i, layer in enumerate(man["layers"]):
    r = urllib.request.urlopen(urllib.request.Request(f"https://public.ecr.aws/v2/{REPO}/blobs/{layer['digest']}", headers=H))
    with tarfile.open(fileobj=r, mode="r|*") as t:
        for m in t:
            m_ = pat.search(m.name)
            if m_:
                found[m_.group(1)] = m.name
            if re.search(r"torch_neuronx/(_version|_build_config)\.py$|libarchive\.so\.13$|/os-release$", m.name):
                files.add(m.name)
    print(f"layer {i+1}/{len(man['layers'])} done", file=sys.stderr, flush=True)
for k in sorted(found):
    if re.match(r"(torch|torch_neuronx|triton|neuronx_cc|nki|torchtitan|numpy)[-_]", k, re.I):
        print("PKG", k)
for f in sorted(files):
    print("FILE", f)
