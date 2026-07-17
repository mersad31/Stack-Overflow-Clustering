# خوشه‌بندی پروفایل توسعه‌دهندگان Stack Overflow

پروژه پایانی درس داده‌کاوی پیشرفته با استفاده از داده‌های پیمایش توسعه‌دهندگان Stack Overflow در سال ۲۰۲۴.

## خلاصه پروژه

پس از پاک‌سازی 65,437 پاسخ خام، مجموعه تحلیلی شامل 60,023 پاسخ و 328 ویژگی ساخته شد. پنج خانواده الگوریتم خوشه‌بندی در 78 پیکربندی مقایسه شدند. نتیجه نهایی از consensus clustering با دو خوشه به دست آمد:

- خوشه 0: تعداد 45,969 پاسخ‌دهنده با پشته فناوری متمرکزتر
- خوشه 1: تعداد 14,054 پاسخ‌دهنده با پشته فناوری گسترده‌تر
- دقت مدل انتساب روی داده holdout: 96.4٪
- fidelity درخت توضیحی: 98.9٪

نتیجه یک تقسیم اکتشافی بر اساس پهنای پشته فناوری است و نباید به‌عنوان دسته‌بندی ذاتی یا معیار ارزیابی افراد تفسیر شود.

## ساختار پروژه

- `src/stackoverflow_clustering/`: کد پاک‌سازی، مهندسی ویژگی، خوشه‌بندی و ارزیابی
- `scripts/`: اجرای مراحل پروژه و ساخت گزارش‌ها
- `data/processed/`: ماتریس‌ها، embeddingها و برچسب‌های نهایی
- `artifacts/`: مدل‌ها و خلاصه اجرای مراحل
- `reports/`: گزارش‌های فارسی در قالب DOCX/PDF
- `reports/figures/`: نمودارهای استفاده‌شده در گزارش‌ها
- `reports/tables/`: خروجی‌های جدولی تحلیل
- `notebooks/`: notebookهای اجراشده سه فاز
- `dashboard/`: داشبورد Streamlit
- `tests/`: آزمون‌های خودکار

## اجرا
### قبل از اجرا هر دو فایل csv داخل ددیتاست را در root قرار دهید.

فرمان‌ها از ریشه پروژه اجرا می‌شوند:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m scripts.run_all --config config.yaml
.\venv\Scripts\python.exe -m scripts.build_all_deliverables
powershell -ExecutionPolicy Bypass -File scripts\export_reports_pdf.ps1
.\venv\Scripts\python.exe -m scripts.validate_release
.\venv\Scripts\python.exe -m pytest
```

اجرای داشبورد:

```powershell
.\venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

## گزارش‌ها

- report/ `Final_Report_FA.pdf`: گزارش جامع نهایی
- report/ `Phase1_Report_FA.pdf`: آماده‌سازی داده و گرایش خوشه‌ای
- report/ `Phase2_Report_FA.pdf`: مقایسه الگوریتم‌ها
- report/ `Phase3_Report_FA.pdf`: consensus، تفسیر و تحلیل تکمیلی
- report/ `Topic_Report_FA.pdf`: انتخاب موضوع، داده و برنامه تحلیل
