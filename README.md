# danorganizer

Scan your osu! library and automatically organize your 4K osu!mania beatmaps. Powered by the Companella difficulty workflow, this CLI tool evaluates your maps and generates Dan and skillset collections directly in your collection.db.

## Usage

```bash
# Preview classification without modifying collection.db
danorganizer --test --limit 50

# Classify entire library and update collection.db
danorganizer

# Run directly from repository folder
./run.sh
```

## Options

```
Usage: danorganizer [options]

  --songs <path>       Path to osu! Songs folder
  --db <path>          Path to osu! collection.db
  --threads <n>        Worker thread count (default: CPU logical cores)
  --limit <n>          Process at most N maps
  --test, --dry-run    Run evaluation without writing to collection.db
  --simple-names       Output 'Dan 1'..'Dan 10', 'Alpha'..'Kappa' instead of Greek letters
  --purge-dans         Remove existing Dan collections before writing new ones
  -h, --help           Show help
```

## How it works

1. Filters `.osu` headers in memory (`Mode: 3`, `CircleSize: 4`). Non-4K mania files are discarded before running subprocesses.
2. Extracts 10 difficulty metrics per map:
   - 8 skillset ratings from MinaCalc 5.15 (Overall, Stream, Jumpstream, Handstream, Stamina, Jackspeed, Chordjack, Technical).
   - Strain density curve from YAVSRG Interlude (top 2,500 notes).
   - Pattern complexity and density from Sunny.
3. Evaluates the 10 features via ONNX runtime against `dan_model.onnx` to produce a continuous difficulty value ($1.0 \le y \le 20.0$), rounded to the nearest integer Dan level.
4. Caches hashes and classification results in `dan_cache.json`. Subsequent runs only calculate new or modified files.
5. Backs up `collection.db` to `collection.db.bak_YYYYMMDD_HHMMSS`, preserves existing personal collections, and writes updated Dan categories using osu! ULEB128 string serialization.

## Platform support

| Platform | Difficulty Calculator | osu! Path Resolution |
| :--- | :--- | :--- |
| Linux | Native 64-bit ELF binary | Auto-detected (`~/.local/share/osu-wine`, Lutris, Flatpak, wineprefixes) |
| Windows | Native 64-bit PE executable | Auto-detected (`%LOCALAPPDATA%/osu!`, `%APPDATA%`, system drives) |
| macOS | Fallback execution / wine | Auto-detected (`~/Library/Application Support/osu`) |

## Credits

- [Companella](https://github.com/leyna64/companella) (model training, ML pipeline, Sunny/Interlude difficulty algorithms)
- [MinaCalc](https://github.com/etterna/etterna) (Mina and StepMania/Etterna contributors)
- [YAVSRG](https://github.com/YAVSRG/YAVSRG) (Interlude difficulty model)

## License

MIT
