# Windows 可玩包

发布包包含游戏程序、默认 Hard 模型、启动脚本、规则与许可。玩家解压后双击 `start.cmd` 即可游玩，不需要安装 Rust、Python 或 GPU 环境。

维护者从干净、已提交的源码构建静态 CRT 的 Windows x64 程序：

```powershell
$env:CARGO_TARGET_DIR = 'research/artifacts/builds/portable'
$env:RUSTFLAGS = '-C target-feature=+crt-static'
cargo build --release --bin cabo-server
python scripts/package_release.py --exe research/artifacts/builds/portable/release/cabo-server.exe
```

产物位于 `research/artifacts/releases/`，该目录不随 Git 提交。构建工具核对预训练模型的 SHA256、可执行文件架构与 VC 运行库依赖，并附上源码版本、文件校验和及第三方许可。

预览尚未提交的修改时，先用上述命令重新构建程序，再给打包命令加上 `--preview`。预览包名称含 `preview`，构建信息会明确记录本地修改状态和源码文件校验和，便于核对。

v1.0 的对战表同时写入包内 `README.txt`，对应报告与原始数据位于包内 `benchmarks/`。

源码仓库直接包含 `models/hard-value-v1.bin`；用户从源码运行时同样无需重新训练模型。
