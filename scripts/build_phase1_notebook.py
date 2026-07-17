from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def build(output: Path) -> None:
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (project venv)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.13"},
    }
    notebook.cells = [
        markdown(
            "# فاز اول — آماده‌سازی داده و ارزیابی گرایش به خوشه‌بندی\n\n"
            "**داده:** Stack Overflow Developer Survey 2024  \n"
            "**سؤال:** کشف پروفایل‌های طبیعی توسعه‌دهندگان براساس پشته فناوری، تجربه و بستر کاری.\n\n"
            "این نوت‌بوک خروجی‌های pipeline بازتولیدپذیر فاز اول را مستقیما از artifactهای ثبت‌شده مرور می‌کند."
        ),
        code(
            "from pathlib import Path\n"
            "import json, sys\n"
            "import pandas as pd\n"
            "from IPython.display import display, Image\n\n"
            "ROOT = Path.cwd().resolve()\n"
            "if ROOT.name == 'notebooks': ROOT = ROOT.parent\n"
            "print('Python:', sys.executable)\n"
            "print('Project root:', ROOT)"
        ),
        markdown("## ۱. ورود داده و cohort تحلیلی"),
        code(
            "ingestion = json.loads((ROOT/'artifacts/ingestion_manifest.json').read_text(encoding='utf-8'))\n"
            "cleaning = json.loads((ROOT/'artifacts/cleaning_summary.json').read_text(encoding='utf-8'))\n"
            "pd.DataFrame({\n"
            "  'شاخص': ['ردیف خام','ستون خام','ستون منتخب','ردیف cohort','ردیف حذف‌شده'],\n"
            "  'مقدار': [ingestion['rows'], ingestion['source_columns'], ingestion['selected_columns'], cleaning['rows_cleaned'], cleaning['rows_removed']]\n"
            "})"
        ),
        code(
            "decisions = pd.read_csv(ROOT/'reports/tables/cleaning_decision_log.csv')\n"
            "decisions[['column','issue','action','affected_rows']].head(12)"
        ),
        code("display(Image(filename=ROOT/'reports/figures/phase1_missingness_selected_fields.png', width=850))"),
        markdown(
            "۵٬۴۱۴ پاسخ‌دهنده که در هر هفت حوزه فناوری فاقد پاسخ بودند از cohort تحلیلی کنار گذاشته شدند. "
            "هیچ ResponseId تکراری و هیچ شکست صریح آزمون توجه مشاهده نشد. مقدارهای تجربه با میانه کشور و fallback سراسری برآورد شدند و indicator گمشدگی حفظ شد."
        ),
        markdown("## ۲. مهندسی ویژگی و scaling"),
        code("pd.read_csv(ROOT/'reports/tables/feature_inventory.csv')"),
        code("display(Image(filename=ROOT/'reports/figures/phase1_scaler_comparison.png', width=900))"),
        markdown(
            "RobustScaler برای بلوک عددی انتخاب شد، زیرا تجربه و اندازه سازمان دنباله‌های سنگین و نقاط دورافتاده واقعی دارند. "
            "ویژگی‌های دودویی one-hot/multi-hot بدون center کردن حفظ شدند."
        ),
        code(
            "tech = pd.read_csv(ROOT/'reports/tables/technology_prevalence.csv')\n"
            "tech.sort_values('respondents', ascending=False).head(15)"
        ),
        code("display(Image(filename=ROOT/'reports/figures/phase1_top_technologies.png', width=850))"),
        markdown("## ۳. کاهش بعد"),
        code(
            "reduction = json.loads((ROOT/'artifacts/reduction_manifest.json').read_text(encoding='utf-8'))\n"
            "pd.Series({\n"
            " 'ابعاد PCA': str(reduction['pca_shape']),\n"
            " 'واریانس تجمعی 50 مؤلفه': reduction['pca_cumulative_variance'],\n"
            " 'ابعاد SVD فناوری': str(reduction['svd_shape']),\n"
            " 'واریانس تجمعی SVD': reduction['svd_cumulative_variance'],\n"
            " 'نمونه UMAP': reduction['umap_sample_size'],\n"
            " 'وضعیت UMAP': reduction['umap_status']\n"
            "})"
        ),
        code("display(Image(filename=ROOT/'reports/figures/phase1_pca_explained_variance.png', width=760))"),
        code("display(Image(filename=ROOT/'reports/figures/phase1_umap_density.png', width=760))"),
        markdown(
            "پنجاه مؤلفه PCA برابر ۷۷٫۸٪ واریانس X_full را نگه می‌دارند. این انتخاب مصالحه‌ای میان حفظ اطلاعات و کاهش تمرکز فاصله‌هاست. "
            "UMAP وجود نواحی متراکم و ساختار پیوسته/چندلوبی را نشان می‌دهد؛ بنابراین انتظار خوشه‌های کاملا کروی ساده واقع‌بینانه نیست."
        ),
        markdown("## ۴. گرایش به خوشه‌بندی: Hopkins و VAT"),
        code("pd.read_csv(ROOT/'reports/tables/hopkins_summary.csv').round(4)"),
        code("display(Image(filename=ROOT/'reports/figures/phase1_vat_heatmap.png', width=760))"),
        markdown(
            "میانگین Hopkins در تمام نمایش‌های گزارش‌شده بالاتر از آستانه عملی ۰٫۷ قرار دارد و نمایش PCA قوی‌ترین گرایش را نشان می‌دهد. "
            "VAT گرادیان و چند ناحیه پیوسته را بیش از بلوک‌های کاملا جدا نشان می‌دهد. جمع‌بندی محافظه‌کارانه این است که ساختار غیرتصادفی وجود دارد، "
            "اما هم‌پوشانی، شکل غیرکوژ و تغییر چگالی محتمل است؛ در نتیجه portfolio فاز دوم باید فراتر از K-Means باشد."
        ),
        markdown("## ۵. نتیجه فاز اول\n\nداده آمادگی ورود به مقایسه K-Means، خوشه‌بندی سلسله‌مراتبی، روش‌های چگالی، GMM و Spectral را دارد. تمام ماتریس‌ها، scalerها، reducerها، هش‌ها و جداول در artifactهای پروژه ثبت شده‌اند."),
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    executed = client.execute()
    nbf.write(executed, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="notebooks/01_phase1_eda_executed.ipynb")
    args = parser.parse_args()
    build((ROOT / args.output).resolve())
    print(f"Wrote executed notebook: {args.output}")


if __name__ == "__main__":
    main()
