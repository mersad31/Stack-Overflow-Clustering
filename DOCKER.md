# Docker — Stack Overflow 2024 Clustering

این پوشه‌ی ریشه پروژه شامل فایل‌های زیر برای Docker است:

| فایل | نقش |
|------|-----|
| `Dockerfile` | ایمیج چندمرحله‌ای Python 3.13 با تمام وابستگی‌های `requirements.txt` |
| `.dockerignore` | جلوگیری از ورود کش‌ها، venv و داده‌های سنگین به build context |
| `docker-compose.yml` | سرویس‌های dashboard، phase2، phase3، build-deliverables، validate، pytest |
| `.gitignore` | نادیده گرفتن فایل‌های پایتون/IDE/OS و داده‌های خام سنگین در Git |

## پیش‌نیازها

- Docker Engine ≥ 24.0
- Docker Compose v2 (پلاگین `docker compose`)

## ۱) ساخت ایمیج

```bash
docker build -t so-clustering:latest .
# یا با compose:
docker compose build dashboard
```

> ⚠️ مرحله‌ی build به دلیل کامپایل `hdbscan`, `umap-learn`, `shap`, `scikit-learn` و `pyarrow` ممکن است ۱۰–۲۰ دقیقه طول بکشد. این فقط یک‌بار انجام می‌شود.

## ۲) اجرای داشبورد (پیش‌فرض)

```bash
docker compose up -d
# آدرس: http://localhost:8501
docker compose logs -f dashboard
docker compose down
```

## ۳) اجرای فازهای تحلیل

```bash
# Phase 2 — partitioning + model families
docker compose --profile analysis run --rm phase2

# Phase 3 — consensus clustering + bonus
docker compose --profile analysis run --rm phase3

# ساخت همه‌ی notebookها، نمودارها و گزارش‌های DOCX
docker compose --profile analysis run --rm build-deliverables

# اعتبارسنجی release
docker compose --profile analysis run --rm validate

# اجرای تست‌ها
docker compose --profile analysis run --rm pytest
```

## ۴) اجرای دستی (بدون compose)

```bash
# شل تعاملی داخل کانتینر
docker run --rm -it -v "$PWD/data:/app/data" \
    -v "$PWD/artifacts:/app/artifacts" \
    -v "$PWD/reports:/app/reports" \
    -v "$PWD/src:/app/src" \
    -v "$PWD/scripts:/app/scripts" \
    -v "$PWD/config.yaml:/app/config.yaml:ro" \
    -p 8501:8501 so-clustering bash

# اجرای یک اسکریپت خاص
docker run --rm -v "$PWD:/app" so-clustering \
    python -m scripts.run_phase2 --config config.yaml
```

## ۵) Volumeها

این مسیرها از host روی کانتینر mount می‌شوند تا داده‌های سنگین جدا از ایمیج بمانند:

| Host | Container |وضعیت |
|------|-----------|------|
| `./data` | `/app/data` | خواندن/نوشتن |
| `./artifacts` | `/app/artifacts` | خواندن/نوشتن |
| `./reports` | `/app/reports` | خواندن/نوشتن |
| `./src` | `/app/src` | خواندن/نوشتن (توسعه‌ی زنده) |
| `./scripts` | `/app/scripts` | خواندن/نوشتن |
| `./dashboard` | `/app/dashboard` | خواندن/نوشتن |
| `./notebooks` | `/app/notebooks` | خواندن/نوشتن |
| `./config.yaml` | `/app/config.yaml` | فقط خواندن |

## ۶) عیب‌یابی

- **خطای `ModuleNotFoundError: stackoverflow_clustering`** — مطمئن شوید پوشه‌ی `src/stackoverflow_clustering/` روی host وجود دارد و حاوی کد پکیج است.
- **خطای `FileNotFoundError` در dashboard** — داده‌های `data/processed/`, `reports/tables/`, `artifacts/` هنوز ساخته نشده‌اند. اول `phase2` و `phase3` را اجرا کنید.
- **build کند** — cache layer با `COPY requirements.txt` جدا شده؛ وقتی کد تغییر می‌کند ولی requirements نه، build سریع است.
- **OOM در phase2/phase3** — `config.yaml` را ویرایش و `*_sample_size` را کاهش دهید یا `--memory` به `docker run` اضافه کنید.
- **پورت اشغال** — در `docker-compose.yml` پورت `8501` را به مثلاً `8601:8501` تغییر دهید.

## ۷) متغیرهای محیطی مفید

| متغیر | پیش‌فرض | توضیح |
|-------|---------|-------|
| `PYTHONPATH` | `/app:/app/src` | مسیر ایمپورت پکیج |
| `STREAMLIT_SERVER_PORT` | `8501` | پورت Streamlit |
| `STREAMLIT_SERVER_HEADLESS` | `true` | بدون باز شدن مرورگر |
| `MPLCONFIGDIR` | `/tmp/matplotlib` | قابل‌نوشتن بودن کش matplotlib |
