param(
    [string]$EnvFile = ".env",
    [switch]$RequireCuda
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Check {
    param([string]$Status, [string]$Message)
    Write-Host ("[{0}] {1}" -f $Status, $Message)
}

function Load-DotEnv {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Env file not found: $Path"
    }

    Get-Content -LiteralPath $Path | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) { return }
        $parts = $line -split "=", 2
        if ($parts.Length -ne 2) { return }

        $key = $parts[0].Trim()
        $value = $parts[1].Trim()
        if ($value.StartsWith("'") -and $value.EndsWith("'")) {
            $value = $value.Substring(1, $value.Length - 2)
        } elseif ($value.StartsWith('"') -and $value.EndsWith('"')) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        if ($key) {
            Set-Item -Path ("Env:{0}" -f $key) -Value $value
        }
    }
}

function Get-ModelExpectedDimensions {
    param([string]$Model)
    if (-not $Model) { return $null }

    $m = $Model.ToLowerInvariant()
    if ($m -like "*nomic*") { return 768 }
    if ($m -eq "text-embedding-3-small") { return 1536 }
    if ($m -eq "text-embedding-3-large") { return 3072 }
    if ($m -eq "text-embedding-ada-002") { return 1536 }
    return $null
}

Load-DotEnv -Path $EnvFile

$pythonExe = ".\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $pythonExe = "python"
}

Write-Host "Running preflight checks..."
Write-Host "Using Python: $pythonExe"
Write-Host "Using env file: $EnvFile"

$embeddingModel = $env:EMBEDDING_MODEL
$embeddingDims = $env:EMBEDDING_DIMENSIONS
if (-not $embeddingModel) { throw "EMBEDDING_MODEL is not set in $EnvFile" }
if (-not $embeddingDims) { throw "EMBEDDING_DIMENSIONS is not set in $EnvFile" }

[int]$embeddingDimsInt = $embeddingDims
$modelExpected = Get-ModelExpectedDimensions -Model $embeddingModel
if ($null -ne $modelExpected -and $modelExpected -ne $embeddingDimsInt) {
    throw "Env mismatch: model '$embeddingModel' typically outputs $modelExpected dims, but EMBEDDING_DIMENSIONS=$embeddingDimsInt."
}
Write-Check -Status "OK" -Message "Embedding config: model=$embeddingModel, dimensions=$embeddingDimsInt"

$torchJson = & $pythonExe -c "import json,torch; print(json.dumps({'torch': torch.__version__, 'cuda_compiled': torch.version.cuda, 'cuda_available': torch.cuda.is_available(), 'device': (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)}))"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to inspect torch/CUDA using $pythonExe."
}

$torchInfo = $torchJson | ConvertFrom-Json
if ($torchInfo.cuda_available) {
    Write-Check -Status "OK" -Message ("CUDA detected: {0} (torch={1}, cuda={2})" -f $torchInfo.device, $torchInfo.torch, $torchInfo.cuda_compiled)
} else {
    $msg = "CUDA not available (torch=$($torchInfo.torch), cuda=$($torchInfo.cuda_compiled))."
    if ($RequireCuda) { throw $msg }
    Write-Check -Status "WARN" -Message $msg
}

$dbJson = @'
import asyncio
import json
import os
import re
import asyncpg

async def main():
    url = os.getenv("DATABASE_URL")
    out = {"database_url_set": bool(url), "db_dim": None, "error": None}
    if not url:
        print(json.dumps(out))
        return

    conn = None
    try:
        conn = await asyncpg.connect(url)
        row = await conn.fetchrow(
            """
            SELECT format_type(a.atttypid, a.atttypmod) AS embedding_type
            FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = current_schema()
              AND c.relname = 'chunks'
              AND a.attname = 'embedding'
              AND NOT a.attisdropped
            """
        )
        embedding_type = row["embedding_type"] if row else None
        match = re.match(r"vector\((\d+)\)", embedding_type or "")
        out["db_dim"] = int(match.group(1)) if match else None
    except Exception as e:
        out["error"] = str(e)
    finally:
        if conn is not None:
            await conn.close()

    print(json.dumps(out))

asyncio.run(main())
'@ | & $pythonExe -

if ($LASTEXITCODE -ne 0) {
    throw "Failed to query database embedding dimensions."
}

$dbInfo = $dbJson | ConvertFrom-Json
if ($dbInfo.error) { throw ("Database check failed: {0}" -f $dbInfo.error) }
if (-not $dbInfo.database_url_set) { throw "DATABASE_URL is not set in $EnvFile." }
if ($null -eq $dbInfo.db_dim) { throw "Could not read chunks.embedding vector dimension from database." }
if ([int]$dbInfo.db_dim -ne $embeddingDimsInt) {
    throw "DB/env mismatch: chunks.embedding is $($dbInfo.db_dim) but EMBEDDING_DIMENSIONS is $embeddingDimsInt."
}
Write-Check -Status "OK" -Message ("Database vector dimension: {0}" -f $dbInfo.db_dim)

Write-Host ""
Write-Host "Preflight passed."
