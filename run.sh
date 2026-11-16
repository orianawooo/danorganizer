#!/usr/bin/env bash
set -e

if [ -d "$HOME/.dotnet" ]; then
    export DOTNET_ROOT="$HOME/.dotnet"
    export PATH="$HOME/.dotnet:$PATH"
fi

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
    DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
BIN="$SCRIPT_DIR/bin/Release/net8.0/danorganizer"

if [ ! -f "$BIN" ]; then
    DOTNET_CMD="dotnet"
    if ! command -v dotnet &>/dev/null; then
        if [ -x "$HOME/.dotnet/dotnet" ]; then
            DOTNET_CMD="$HOME/.dotnet/dotnet"
        else
            echo "Error: .NET 8.0 SDK not found in PATH or ~/.dotnet" >&2
            exit 1
        fi
    fi
    "$DOTNET_CMD" build -c Release "$SCRIPT_DIR/danorganizer.csproj"
fi

exec "$BIN" "$@"
