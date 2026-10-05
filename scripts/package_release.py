"""Build a Windows playable archive from a clean source revision and static-CRT executable."""
import argparse
import hashlib
import json
import struct
import subprocess
import zipfile
from pathlib import Path


MODEL_SHA256 = "f41dedda1ab25df3b0f7692807724831a89dffdd9118fa05d55f8fb6ecd96fc8"


def windows_imports(raw):
    if raw[:2] != b"MZ":
        raise ValueError("expected Windows executable")
    pe = struct.unpack_from("<I", raw, 0x3C)[0]
    if raw[pe:pe + 4] != b"PE\0\0" or struct.unpack_from("<H", raw, pe + 4)[0] != 0x8664:
        raise ValueError("expected Windows x64 executable")
    count, optional_size = struct.unpack_from("<H", raw, pe + 6)[0], struct.unpack_from("<H", raw, pe + 20)[0]
    optional = pe + 24
    if struct.unpack_from("<H", raw, optional)[0] != 0x20B:
        raise ValueError("expected PE32+")
    sections = []
    for i in range(count):
        pos = optional + optional_size + 40 * i
        size, address, disk_size, disk_pos = struct.unpack_from("<IIII", raw, pos + 8)
        sections.append((address, max(size, disk_size), disk_pos))

    def offset(address):
        for start, size, disk_pos in sections:
            if start <= address < start + size:
                return disk_pos + address - start
        raise ValueError("invalid import address")

    address = struct.unpack_from("<I", raw, optional + 120)[0]
    if not address:
        return []
    pos, imports = offset(address), []
    while any(raw[pos:pos + 20]):
        name = offset(struct.unpack_from("<I", raw, pos + 12)[0])
        imports.append(raw[name:raw.index(0, name)].decode("ascii"))
        pos += 20
    if any(name.lower().startswith(("vcruntime", "msvcp")) for name in imports):
        raise ValueError("build with static CRT; portable archive must not require VC runtime installation")
    return imports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True)
    parser.add_argument("--out", default="research/artifacts/releases/CaboGame-Windows-x64.zip")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip():
        raise ValueError("commit the source before packaging")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    output = (root / args.out).resolve()
    if not output.is_relative_to(root / "research/artifacts") or output.exists():
        raise ValueError("choose a new output inside research/artifacts")
    executable = (root / args.exe).read_bytes()
    imports = windows_imports(executable)
    model = (root / "models/hard-value-v1.bin").read_bytes()
    if hashlib.sha256(model).hexdigest() != MODEL_SHA256:
        raise ValueError("model is not the evaluated champion")
    payload = {"cabo-server.exe": executable, "models/hard-value-v1.bin": model}
    for name in ["LICENSE", "THIRD_PARTY_NOTICES.md", "models/README.md", "rule.md"]:
        payload[name] = (root / name).read_bytes()
    payload["start.cmd"] = (
        '@echo off\r\nsetlocal\r\ncd /d "%~dp0"\r\n'
        'if not defined PORT set PORT=8080\r\n'
        'echo CaboGame: http://localhost:%PORT%\r\n'
        'start "" "http://localhost:%PORT%"\r\n'
        'cabo-server.exe\r\npause\r\n'
    ).encode("ascii")
    payload["README.txt"] = (
        "CaboGame Windows x64\n\n"
        "解压到任意目录，双击 start.cmd，浏览器访问 http://localhost:8080。\n"
        "默认 Hard 模型已包含，无需 Rust、Python、GPU 或训练。\n"
        "创建房间后可以添加 AI；朋友可通过主机的局域网地址加入。\n"
        "如网页打开过早，请刷新。按 Ctrl+C 停止服务。\n"
        "换端口：在终端设置 PORT 后运行 cabo-server.exe。\n\n"
        f"源码版本：{revision}\n"
        "项目：https://github.com/huixiong806/CaboGame\n"
        "许可：MIT；第三方许可见 THIRD_PARTY_NOTICES.md 与 licenses/。\n"
    ).encode("utf-8")
    metadata = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--locked", "--format-version", "1"], cwd=root, text=True, encoding="utf-8"
    ))
    dependencies = []
    for package in metadata["packages"]:
        if package["name"] == "cabo":
            continue
        dependencies.append({key: package.get(key) for key in ["name", "version", "license", "repository"]})
        directory = Path(package["manifest_path"]).parent
        for path in directory.iterdir():
            if path.is_file() and path.name.upper().startswith(("LICENSE", "COPYING", "NOTICE")):
                payload[f"licenses/{package['name']}-{package['version']}/{path.name}"] = path.read_bytes()
    payload["licenses/dependencies.json"] = json.dumps(dependencies, ensure_ascii=False, indent=2).encode("utf-8")
    payload["BUILD.json"] = json.dumps({
        "source_commit": revision, "platform": "windows-x86_64", "imports": imports,
        "model_sha256": MODEL_SHA256, "binary_sha256": hashlib.sha256(executable).hexdigest(),
    }, indent=2).encode("utf-8")
    payload["SHA256SUMS"] = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in sorted(payload.items())
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(payload.items()):
            archive.writestr("CaboGame/" + name, data)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".zip.sha256").write_text(f"{digest}  {output.name}\n", encoding="ascii")
    print(json.dumps({"archive": str(output), "bytes": output.stat().st_size, "sha256": digest,
                      "source_commit": revision, "imports": imports}, ensure_ascii=False))


if __name__ == "__main__":
    main()
