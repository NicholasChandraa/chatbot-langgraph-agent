# Cara Install pgvector di Windows (PostgreSQL 17)

Panduan ini menjelaskan cara install pgvector extension untuk PostgreSQL 17 di Windows menggunakan pre-built binary, karena compile manual sering gagal di Windows.

## Masalah yang Sering Terjadi

Saat mencoba compile pgvector dari source di Windows, sering muncul error:
```
error C2196: case value '4' already used
```

Ini adalah known issue compatibility antara pgvector dan Visual Studio compiler di Windows.

## Solusi: Download Pre-built Binary

Gunakan pre-built binary dari repository **andreiramani/pgvector_pgsql_windows** yang sudah dikompilasi khusus untuk Windows.

## Langkah-langkah Instalasi

### 1. Download Pre-built Binary

- Buka: https://github.com/andreiramani/pgvector_pgsql_windows/releases
- Cari release untuk **PostgreSQL 17** (release Sep 16, 2024)
- Download file `.zip` untuk PostgreSQL 17

### 2. Extract File

Extract file `.zip` yang sudah didownload ke folder temporary (misalnya: `C:\temp\pgvector`)

### 3. Copy Files ke PostgreSQL Directory

Buka **PowerShell as Administrator**, lalu jalankan:

```powershell
# Sesuaikan path ke folder extracted
$extractPath = "C:\temp\pgvector"

# Copy vector.dll ke folder lib PostgreSQL
Copy-Item "$extractPath\lib\vector.dll" "C:\Program Files\PostgreSQL\17\lib\"

# Copy extension files (.sql dan .control) ke folder extension
Copy-Item "$extractPath\share\extension\vector*" "C:\Program Files\PostgreSQL\17\share\extension\"
```

**Catatan:** Pastikan path `$extractPath` sesuai dengan lokasi Anda extract file.

### 4. Enable Extension di Database

Setelah files berhasil dicopy, enable extension di database:

```powershell
# Jalankan command ini (sesuaikan port, user, dan database name)
psql -h localhost -p 5433 -U postgres -d chatbot_chat_db2 -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

Jika berhasil, akan muncul output:
```
CREATE EXTENSION
```

### 5. Verifikasi Instalasi

Cek apakah pgvector sudah terinstall:

```powershell
psql -h localhost -p 5433 -U postgres -d chatbot_chat_db2 -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

Atau cek versi pgvector:

```sql
SELECT vector_version();
```

## Troubleshooting

### Error: Permission Denied saat Copy Files

**Solusi:** Jalankan PowerShell as Administrator

### Error: psql command not found

**Solusi:** Tambahkan PostgreSQL ke PATH:
```powershell
$env:Path += ";C:\Program Files\PostgreSQL\17\bin"
```

Atau gunakan full path:
```powershell
& "C:\Program Files\PostgreSQL\17\bin\psql.exe" -h localhost -p 5433 -U postgres -d chatbot_chat_db2 -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Extension Sudah Ada

Jika muncul error `extension "vector" already exists`, berarti extension sudah terinstall sebelumnya. Tidak perlu install ulang.

## Catatan Penting

- **Port Database:** Sesuaikan port (`5433`) dengan konfigurasi PostgreSQL Anda
- **Database Name:** Sesuaikan nama database (`chatbot_chat_db2`) dengan database Anda
- **PostgreSQL Version:** Guide ini untuk PostgreSQL 17. Untuk versi lain (16, 18), download release yang sesuai

## Referensi

- Repository: https://github.com/andreiramani/pgvector_pgsql_windows
- pgvector Official: https://github.com/pgvector/pgvector

## Kenapa Tidak Compile Manual?

Compile manual pgvector di Windows memerlukan:
- Visual Studio dengan C++ support
- Developer Command Prompt for VS
- Setup environment yang rumit
- Sering gagal dengan compatibility errors

Pre-built binary **jauh lebih mudah, cepat, dan reliable** untuk Windows.
