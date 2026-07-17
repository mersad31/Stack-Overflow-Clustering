from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stackoverflow_clustering.config import load_config
from stackoverflow_clustering.reporting import PersianReport, add_project_cover


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: float, digits: int = 1) -> str:
    return f"{100 * float(value):.{digits}f}٪"


def build() -> Path:
    config = load_config(ROOT / "config.yaml")
    artifacts = ROOT / "artifacts"
    tables = ROOT / "reports" / "tables"
    figures = ROOT / "reports" / "figures"

    phase1 = _json(artifacts / "phase1_run_summary.json")
    phase2 = _json(artifacts / "phase2_run_summary.json")
    phase3 = _json(artifacts / "phase3_run_summary.json")
    bonus = _json(artifacts / "bonus_summary.json")

    cleaning_log = pd.read_csv(tables / "cleaning_decision_log.csv")
    feature_inventory = pd.read_csv(tables / "feature_inventory.csv")
    hopkins = pd.read_csv(tables / "hopkins_summary.csv")
    partitioning = pd.read_csv(tables / "phase2_partitioning_scores.csv")
    family_scores = pd.read_csv(tables / "phase2_model_family_scores.csv")
    bootstrap_k = pd.read_csv(tables / "phase2_bootstrap_k_stability.csv")
    proxies = pd.read_csv(tables / "phase2_proxy_label_metrics.csv")
    consensus_selection = pd.read_csv(tables / "phase3_consensus_k_selection.csv")
    profiles = pd.read_csv(tables / "phase3_cluster_profiles.csv")
    technologies = pd.read_csv(tables / "phase3_cluster_top_technologies.csv")
    exemplars = pd.read_csv(tables / "phase3_exemplars_boundaries.csv")
    shap_values = pd.read_csv(tables / "phase3_feature_importance_shap.csv")
    downstream = pd.read_csv(tables / "phase3_downstream_compensation.csv")
    composition = pd.read_csv(tables / "phase3_composition_audit.csv")
    sensitivity = pd.read_csv(tables / "phase3_sensitivity.csv")
    drift = pd.read_csv(tables / "phase3_drift_baseline.csv")
    anomalies = pd.read_csv(tables / "phase3_cluster_anomalies.csv")
    nmf = pd.read_csv(tables / "bonus_nmf_scores.csv")
    confidence = pd.read_csv(tables / "bonus_metric_confidence_intervals.csv")
    permutation = pd.read_csv(tables / "bonus_permutation_tests.csv")
    split = pd.read_csv(tables / "bonus_split_stability.csv")

    p1_ingestion = phase1["ingestion"]
    p1_cleaning = phase1["cleaning"]
    p1_features = phase1["features"]
    p1_reduction = phase1["reduction"]
    p2_partitioning = phase2["partitioning"]
    p2_families = phase2["model_families"]
    selected_consensus = consensus_selection.loc[
        consensus_selection["k"] == phase3["consensus_k"]
    ].iloc[0]

    report = PersianReport("کشف و تفسیر پروفایل‌های توسعه‌دهندگان در پیمایش Stack Overflow 2024")
    add_project_cover(report, "گزارش جامع نهایی پروژه داده‌کاوی پیشرفته")
    report.add_toc()

    report.heading("چکیده", 1)
    report.paragraph(
        "این پروژه یک مطالعه کامل خوشه‌بندی بدون‌نظارت روی پیمایش توسعه‌دهندگان Stack Overflow در سال ۲۰۲۴ است. "
        "پرسش اصلی آن است که آیا از ترکیب پشته فناوری، تجربه حرفه‌ای و بستر کاری می‌توان پروفایل‌های طبیعی، "
        "پایدار و قابل‌تفسیر استخراج کرد و این پروفایل‌ها تا چه اندازه برای تحلیل‌های بعدی ارزش دارند. "
        f"از {p1_ingestion['rows']:,} پاسخ خام و {p1_ingestion['source_columns']} ستون، پس از کنترل schema، "
        f"تحلیل گمشدگی و حذف {p1_cleaning['rows_removed']:,} پاسخ فاقد هرگونه اطلاعات فناوری، cohort نهایی "
        f"{p1_cleaning['rows_cleaned']:,} نفری ساخته شد. نمایش کامل شامل {p1_features['representations']['X_full'][1]} "
        f"ویژگی و نمایش فناوری شامل {p1_features['representations']['X_tech_stack'][1]} ویژگی است. RobustScaler انتخاب شد "
        f"و PCA با ۵۰ مؤلفه، {_pct(p1_reduction['pca_cumulative_variance'], 2)} واریانس را حفظ کرد."
    )
    report.paragraph(
        f"در فاز دوم، {p2_families['configurations_evaluated']} پیکربندی از پنج خانواده partitioning، سلسله‌مراتبی، "
        "چگالی‌محور، مدل‌مبنا و Spectral با معیارهای داخلی، متغیرهای کمکی، پایداری seed، bootstrap و توافق زوجی مقایسه شدند. "
        "شواهد تعداد خوشه هم‌جهت نبودند: silhouette و جمع رتبه‌ها k=2، Kneedle مقدار 6، Gap 1-SE مقدار 7 و بهترین "
        "پایداری bootstrap مقدار 3 را پیشنهاد کردند. در فاز سوم، consensus مبتنی بر co-association مقدار k=2 را برگزید "
        f"و دو پروفایل با اندازه‌های {phase3['cluster_sizes']['0']:,} و {phase3['cluster_sizes']['1']:,} ساخت. "
        f"دقت انتساب روی holdout برابر {_pct(phase3['validation_accuracy'])} و fidelity درخت توضیحی "
        f"{_pct(phase3['explanation_tree_validation_accuracy'])} بود. تفاوت غالب دو خوشه پهنای پشته فناوری با میانه‌های "
        "۱۴ و ۳۲ فناوری است. نتیجه باید یک طیف اکتشافی تلقی شود، نه دو نوع ذاتی از انسان‌ها یا مبنای تصمیم‌گیری فردی."
    )
    report.callout(
        "جمع‌بندی اصلی",
        "در داده ساختاری غیرتصادفی و قابل‌توضیح دیده شد، اما هم‌پوشانی خوشه‌ها، اختلاف معیارهای k، حساسیت به scaler و بهترنشدن مدل‌های "
        "جبران خدمت شرطی بر خوشه نشان می‌دهند که اعتبار نتیجه توصیفی و فرضیه‌ساز است، نه علی یا قطعی.",
    )
    report.table(
        ["شاخص", "نتیجه نهایی"],
        [
            ["پاسخ خام / cohort نهایی", f"{p1_ingestion['rows']:,} / {p1_cleaning['rows_cleaned']:,}"],
            ["نمایش اصلی", f"{p1_features['representations']['X_full'][1]} ویژگی؛ PCA-50"],
            ["پیکربندی‌های فاز دوم", p2_families["configurations_evaluated"]],
            ["روش و k نهایی", f"Consensus؛ k={phase3['consensus_k']}"],
            ["اندازه خوشه‌ها", f"{phase3['cluster_sizes']['0']:,} / {phase3['cluster_sizes']['1']:,}"],
            ["دقت انتساب", _pct(phase3["validation_accuracy"])],
            ["امتیاز افزوده هدف", f"{bonus['points_targeted']} امتیاز"],
        ],
    )

    report.heading("۱. مقدمه و تعریف مسئله", 1)
    report.heading("۱.۱. انگیزه پژوهش", 2)
    report.paragraph(
        "پیمایش Stack Overflow تصویری پربعد از تجربه، نقش، محیط کار و فناوری‌های توسعه‌دهندگان فراهم می‌کند. "
        "گزارش‌های توصیفی معمولا هر متغیر را جداگانه بررسی می‌کنند، در حالی که شخصیت فنی یک توسعه‌دهنده حاصل ترکیب هم‌زمان "
        "زبان‌ها، پایگاه‌های داده، پلتفرم‌ها، ابزارها، سابقه و زمینه شغلی است. خوشه‌بندی امکان می‌دهد الگوهای مشترک بدون "
        "تعریف برچسب از پیش تعیین‌شده کشف شوند؛ با این حال نتیجه فقط زمانی معتبر است که وجود ساختار، پایداری، حساسیت و "
        "قابلیت تفسیر آن جداگانه آزموده شوند."
    )
    report.heading("۱.۲. سؤال و اهداف پژوهش", 2)
    report.callout(
        "سؤال مرکزی",
        "چه پروفایل‌های طبیعی توسعه‌دهندگان از الگوهای مشترک پشته فناوری، تجربه حرفه‌ای و بستر کاری در پیمایش "
        "Stack Overflow 2024 پدیدار می‌شوند، و این پروفایل‌ها چه ارتباطی با نقش، تحصیلات و پذیرش ابزارهای هوش مصنوعی دارند؟",
    )
    report.paragraph("اهداف عملیاتی پژوهش عبارت‌اند از:")
    report.bullet("ساخت pipeline بازتولیدپذیر برای ورود، پاک‌سازی، مهندسی ویژگی، کاهش بعد و خوشه‌بندی.")
    report.bullet("آزمون وجود گرایش خوشه‌ای پیش از انتخاب الگوریتم یا تعداد خوشه.")
    report.bullet("مقایسه منصفانه چند خانواده الگوریتم و چند معیار انتخاب k روی نمونه‌ها و نمایش‌های ثابت.")
    report.bullet("سنجش پایداری seed و bootstrap، توافق الگوریتم‌ها و حساسیت به پیش‌پردازش.")
    report.bullet("تفسیر انسانی خوشه‌ها با پروفایل، فناوری‌های شاخص، exemplar، درخت تصمیم و SHAP.")
    report.bullet("بررسی ارزش پایین‌دستی، ناهنجاری، composition و محدودیت‌های اخلاقی استفاده از نتیجه.")
    report.heading("۱.۳. دامنه و ادعا", 2)
    report.paragraph(
        "این پژوهش اکتشافی است. DevType، EdLevel، AISelect و ConvertedCompYearly از فضای clustering کنار گذاشته شدند تا "
        "ارزیابی دوری ایجاد نشود؛ این متغیرها فقط برای تفسیر یا تحلیل پایین‌دستی استفاده می‌شوند. بنابراین NMI یا سایر معیارهای خارجی "
        "نسبت به آن‌ها معیار انطباق با یک برچسب مرجع نیست. همچنین نام خوشه‌ها توصیفی است و نباید برای رتبه‌بندی، استخدام یا ارزیابی افراد استفاده شود."
    )

    report.heading("۲. داده، منشأ و ملاحظات اخلاقی", 1)
    report.heading("۲.۱. منبع داده", 2)
    report.paragraph(
        "منبع اصلی Stack Overflow Developer Survey 2024 است. فایل survey_results_public.csv پاسخ‌های عمومی و فایل "
        "survey_results_schema.csv متن پرسش‌ها و schema رسمی را نگه می‌دارد. ورودی اصلی با خواندن chunked ده‌هزارردیفی پردازش "
        "و هش SHA-256 هر فایل ثبت شد. از ResponseId فقط برای ردیابی ردیف در artifactها استفاده شده و هیچ تلاش برای شناسایی هویت افراد انجام نشده است."
    )
    report.table(
        ["مولفه داده", "مقدار"],
        [
            ["تعداد پاسخ خام", f"{p1_ingestion['rows']:,}"],
            ["تعداد ستون خام", p1_ingestion["source_columns"]],
            ["ستون‌های منتخب pipeline", p1_ingestion["selected_columns"]],
            ["اندازه chunk", f"{p1_ingestion['chunk_size']:,}"],
            ["cohort نهایی", f"{p1_cleaning['rows_cleaned']:,}"],
            ["seed پایه", config["project"]["random_state"]],
        ],
    )
    report.heading("۲.۲. متغیرهای تحلیلی", 2)
    report.paragraph(
        "ورودی خوشه‌بندی شامل سه سنجه تجربه، اندازه سازمان، وضعیت اشتغال، دورکاری، کشور و هفت خانواده فناوری است: "
        "زبان برنامه‌نویسی، پایگاه داده، پلتفرم، چارچوب وب، فناوری embedded، ابزار و سایر فناوری‌ها. نقش، تحصیلات، پذیرش AI و "
        "جبران خدمت برای تفسیر نگه داشته شدند. این جداسازی مانع از آن می‌شود که خوشه‌ها مستقیما نسخه‌ای از یک برچسب شناخته‌شده باشند."
    )
    report.heading("۲.۳. اخلاق، سوگیری و محدودیت نمونه", 2)
    report.paragraph(
        "داده خوداظهاری و حاصل مشارکت داوطلبانه است؛ بنابراین نماینده کامل تمام توسعه‌دهندگان جهان نیست. گمشدگی متغیرها، "
        "تفاوت دسترسی منطقه‌ای به Stack Overflow و تفاوت در تفسیر پرسش‌ها می‌تواند ترکیب cohort را تغییر دهد. تحلیل composition "
        "برای کشور، تحصیلات، دورکاری و پذیرش AI انجام شد، اما نتیجه توصیفی آن نباید به ادعای برابری یا تبعیض علی تبدیل شود."
    )

    report.heading("۳. آماده‌سازی داده و مهندسی ویژگی", 1)
    report.heading("۳.۱. کنترل کیفیت و پاک‌سازی", 2)
    report.paragraph(
        f"pipeline ابتدا schema ستون‌های موردنیاز را بررسی کرد. شناسه تکراری و شکست صریح آزمون توجه مشاهده نشد. "
        f"تعداد {p1_cleaning['all_technology_domains_missing_removed']:,} ردیف که در تمام حوزه‌های فناوری فاقد پاسخ بودند حذف شدند؛ "
        f"در نتیجه {p1_cleaning['rows_cleaned']:,} رکورد باقی ماند. تعداد {p1_cleaning['rare_country_categories_grouped']} کشور "
        f"با فراوانی کمتر از {p1_cleaning['rare_country_threshold']} در گروه Other ادغام شد تا ابعاد one-hot و ناپایداری دسته‌های کم‌نمونه کنترل شود."
    )
    report.paragraph(
        "عبارت‌های متنی تجربه مانند «کمتر از یک سال» و «بیش از ۵۰ سال» به مقادیر عددی ۰٫۵ و ۵۱ تبدیل شدند. گمشدگی تجربه "
        "ابتدا با میانه کشور و سپس با میانه سراسری تکمیل شد و indicator گمشدگی نیز حفظ گردید. OrgSize به midpoint بازه نگاشت "
        "و log1p شد. ConvertedCompYearly فقط در تحلیل پایین‌دستی استفاده شد و در فضای clustering حضور ندارد."
    )
    important_cleaning = cleaning_log.sort_values("affected_rows", ascending=False).head(12)
    report.table(
        ["ستون", "مسئله", "اقدام", "ردیف متأثر"],
        important_cleaning[["column", "issue", "action", "affected_rows"]].itertuples(index=False, name=None),
    )
    report.figure(figures / "phase1_missingness_selected_fields.png", "شکل ۱ — الگوی گمشدگی ستون‌های منتخب پیش از پاک‌سازی")
    report.heading("۳.۲. ساخت نمایش ویژگی", 2)
    report.paragraph(
        f"هفت ستون چندبرچسبی فناوری به {p1_features['technology_features']} شاخص دودویی تبدیل شدند و TechnologyBreadth "
        f"تعداد فناوری‌های هر پاسخ‌دهنده را ثبت کرد. Employment به {p1_features['employment_features']} ویژگی چندبرچسبی و Country و "
        "RemoteWork به one-hot تبدیل شدند. ویژگی‌های عددی مشتق‌شده شامل تجربه توافقی، اختلاف سنجه‌های تجربه، نسبت تجربه حرفه‌ای، "
        "اندازه سازمان لگاریتمی، indicatorهای گمشدگی و breadth حوزه‌ها هستند."
    )
    report.table(
        ["بلوک ویژگی", "تعداد ویژگی"],
        feature_inventory[["block", "features"]].itertuples(index=False, name=None),
    )
    report.figure(figures / "phase1_top_technologies.png", "شکل ۲ — فناوری‌های پرتکرار در cohort نهایی")
    report.heading("۳.۳. مقیاس‌بندی", 2)
    report.paragraph(
        "StandardScaler، MinMaxScaler و RobustScaler مقایسه شدند. ویژگی‌های تجربه و اندازه سازمان دنباله‌های بلند و پرت‌های واقعی "
        "دارند؛ ازاین‌رو RobustScaler که بر میانه و IQR متکی است به‌عنوان مسیر اصلی انتخاب شد. این انتخاب به معنی برتری مطلق نیست؛ "
        "در فاز سوم حساسیت برچسب نهایی به هر سه scaler با ARI سنجیده شد."
    )
    report.figure(figures / "phase1_scaler_comparison.png", "شکل ۳ — اثر سه روش مقیاس‌بندی بر هندسه ویژگی‌های عددی")

    report.heading("۴. کاهش بعد، فاصله و گرایش خوشه‌ای", 1)
    report.heading("۴.۱. PCA، SVD و UMAP", 2)
    report.paragraph(
        f"PCA روی نمایش کامل Robust-scaled اجرا شد. ۵۰ مؤلفه {_pct(p1_reduction['pca_cumulative_variance'], 2)} واریانس را حفظ "
        f"کرد و فضای اصلی الگوریتم‌های فاصله‌محور شد. Truncated SVD روی نمایش sparse فناوری نیز {_pct(p1_reduction['svd_cumulative_variance'], 2)} "
        f"واریانس را نگه داشت. UMAP روی نمونه ثابت {p1_reduction['umap_sample_size']:,}تایی صرفا برای تشخیص هندسه و هم‌پوشانی استفاده شد، نه برای برازش مدل نهایی."
    )
    report.figure(figures / "phase1_pca_explained_variance.png", "شکل ۴ — واریانس تجمعی PCA و نقطه انتخاب ۵۰ مؤلفه")
    report.figure(figures / "phase1_umap_density.png", "شکل ۵ — چگالی UMAP روی نمونه ثابت؛ ساختار پیوسته و هم‌پوشان")
    report.heading("۴.۲. انتخاب معیار فاصله", 2)
    report.paragraph(
        "Euclidean در فضای PCA معیار اصلی K-Means، Ward و GMM است، زیرا مؤلفه‌ها عددی و متعامدند. برای نمایش فناوری دودویی، "
        "Jaccard مناسب‌تر است چون نبود مشترک فناوری نباید شباهت مثبت تولید کند. تحلیل حساسیت اجراشده بر تعداد مؤلفه‌های PCA و "
        "سه روش مقیاس‌بندی تمرکز داشت؛ Jaccard نیز برای پایداری فناوری میان splitها استفاده شد. انتخاب metric بخشی از فرض مدل است."
    )
    report.heading("۴.۳. Hopkins و VAT", 2)
    hopkins_display = hopkins.copy()
    for column in ["mean", "std", "min", "max"]:
        hopkins_display[column] = hopkins_display[column].map(lambda value: f"{value:.4f}")
    report.table(
        ["نمایش", "میانگین", "انحراف معیار", "کمینه", "بیشینه"],
        hopkins_display[["representation", "mean", "std", "min", "max"]].itertuples(index=False, name=None),
    )
    report.paragraph(
        "همه مقادیر Hopkins از ۰٫۷ بالاترند و فرض تصادفی‌بودن کامل را تضعیف می‌کنند. بااین‌حال VAT بلوک‌های کاملا جدا نشان نمی‌دهد "
        "و UMAP نیز مرزهای نرم دارد. بنابراین داده واجد ساختار است، اما این ساختار الزاما کروی، کاملا جدا یا دارای یک k بدیهی نیست."
    )
    report.figure(figures / "phase1_vat_heatmap.png", "شکل ۶ — ماتریس VAT بازمرتب‌شده روی نمونه هزارنفری PCA")

    report.heading("۵. سبد الگوریتم‌ها و طراحی آزمایش", 1)
    report.paragraph(
        f"فاز دوم روی X_pca با {p2_partitioning['dimensions']} بعد انجام شد. مسیر partitioning تمام {p2_partitioning['rows']:,} ردیف "
        f"را برای ۱۰ مقدار k بررسی کرد. مقایسه بین‌خانواده‌ای برای جلوگیری از تفاوت نمونه روی یک نمونه ثابت {p2_families['sample_size']:,}تایی "
        f"با seed={p2_families['sample_seed']} انجام شد و {p2_families['configurations_evaluated']} پیکربندی را پوشش داد."
    )
    report.table(
        ["خانواده", "روش‌ها", "نقش در مقایسه"],
        [
            ["Partitioning", "K-Means، MiniBatch K-Means", "خط مبنا، مقیاس‌پذیر و کروی"],
            ["Hierarchical", "single، complete، average، Ward", "بررسی ساختار تو‌در‌تو و linkage"],
            ["Density-based", "DBSCAN، HDBSCAN، OPTICS", "خوشه‌های نامنظم و نویز"],
            ["Model-based", "GMM با چهار covariance", "عضویت نرم و انتخاب AIC/BIC"],
            ["Graph-based", "Spectral nearest-neighbors", "ساختار غیرکوژ روی گراف شباهت"],
        ],
    )
    report.callout(
        "قاعده کنترل کیفیت",
        "برای انتخاب نهایی، هر خوشه باید حداقل ۲٪ نمونه مشترک را شامل شود. این قاعده مانع انتخاب تقسیم‌های ظاهرا پر-silhouette "
        "اما بی‌معنا مانند single-linkage با خوشه تک‌عضوی شد.",
    )

    report.heading("۶. انتخاب k و ارزیابی مقایسه‌ای", 1)
    report.heading("۶.۱. شواهد انتخاب k", 2)
    report.table(
        ["معیار", "k پیشنهادی", "تفسیر"],
        [
            ["Silhouette", 2, "جدایش کلی بهتر در تقسیم درشت"],
            ["جمع رتبه‌های داخلی و پایداری", 2, "مصالحه چندمعیاره"],
            ["Elbow / Kneedle", 6, "نقطه شکست inertia"],
            ["Gap 1-SE", 7, "برتری نسبت به مرجع تصادفی"],
            ["Bootstrap stability", 3, "بالاترین ARI بازنمونه‌گیری"],
            ["GMM BIC", 3, "covariance کامل"],
        ],
    )
    report.paragraph(
        "معیارها پاسخ یکسانی ندادند و این موضوع با ساختار چندمقیاسی و هم‌پوشان داده سازگار است. k=2 یک خلاصه کلی و پایدار از "
        "پهنای پشته فناوری ارائه می‌کند، در حالی که kهای بزرگ‌تر جزئیات بیشتری را جدا می‌کنند. به همین دلیل تصمیم نهایی فقط بر یک معیار متکی نبود."
    )
    report.figure(figures / "phase2_partitioning_selection.png", "شکل ۷ — Elbow، silhouette و معیارهای انتخاب k برای partitioning")
    report.figure(figures / "phase2_gap_statistic.png", "شکل ۸ — Gap Statistic و قاعده یک انحراف معیار")
    report.heading("۶.۲. مقایسه خانواده‌ها", 2)
    best_rows = (
        family_scores.dropna(subset=["silhouette"])
        .query("clusters_found > 1 and minimum_cluster_fraction >= 0.02")
        .sort_values("silhouette", ascending=False)
        .head(10)
        .copy()
    )
    report.table(
        ["خانواده", "الگوریتم", "k", "Silhouette", "DB", "CH", "زمان (ثانیه)"],
        [
            (
                row.family,
                row.algorithm,
                int(row.clusters_found),
                f"{row.silhouette:.3f}",
                f"{row.davies_bouldin:.3f}",
                f"{row.calinski_harabasz:.1f}",
                f"{row.runtime_seconds:.2f}",
            )
            for row in best_rows.itertuples()
        ],
    )
    report.paragraph(
        "بهترین silhouette معتبر بین‌خانواده‌ای به GMM با covariance tied و k=2 رسید؛ Ward با k=2 نیز نتیجه رقابتی داشت. "
        "بهترین BIC متعلق به GMM با covariance کامل و k=3 بود. روش‌های چگالی‌محور در grid مستندشده راه‌حل چندخوشه‌ای معتبر "
        "با سهم نویز و اندازه خوشه قابل‌قبول تولید نکردند؛ این نتیجه با پیوستگی و تغییر چگالی مشاهده‌شده در UMAP سازگار است."
    )
    report.figure(figures / "phase2_family_silhouette_comparison.png", "شکل ۹ — مقایسه silhouette بهترین پیکربندی‌های معتبر هر خانواده")
    report.heading("۶.۳. متغیرهای کمکی، پایداری و توافق", 2)
    report.table(
        ["Proxy تفسیری", "NMI", "تعداد دسته", "محدودیت"],
        [
            (row.proxy, f"{row.normalized_mutual_information:.4f}", int(row.categories), row.interpretation)
            for row in proxies.itertuples()
        ],
    )
    report.paragraph(
        f"برای مرجع MiniBatch K-Means با k=2، میانگین ARI بیست seed برابر {p2_partitioning['seed_stability']['mean']:.3f} و "
        f"میانگین پایداری بیست bootstrap برابر {p2_partitioning['bootstrap_stability']['mean']:.3f} بود. این مقادیر پایداری متوسط را "
        "نشان می‌دهند و ضرورت تجمیع چند الگوریتم در consensus را تقویت می‌کنند. NMI پایین متغیرهای کمکی نیز نشان می‌دهد خوشه‌ها صرفا "
        "بازسازی نقش یا تحصیلات نیستند."
    )
    report.figure(figures / "phase2_bootstrap_k_stability.png", "شکل ۱۰ — پایداری bootstrap برای مقادیر مختلف k")
    report.figure(figures / "phase2_algorithm_agreement.png", "شکل ۱۱ — توافق زوجی الگوریتم‌های نامزد با ARI")

    report.heading("۷. مسیر پیشرفته اصلی: Consensus Clustering", 1)
    report.heading("۷.۱. ساخت ماتریس co-association", 2)
    report.paragraph(
        "چهار نامزد K-Means، Ward، GMM-full-k3 و Spectral-k3 روی نمونه مشترک به ماتریس co-association تبدیل شدند. هر درایه "
        "نسبت مدل‌هایی است که یک زوج را هم‌خوشه می‌دانند. سپس Agglomerative با فاصله یک منهای co-association برای kهای ۲ تا ۶ "
        "اجرا شد. این روش وابستگی به یک هندسه یا تابع هدف منفرد را کاهش می‌دهد، هرچند کیفیت آن همچنان به تنوع و کیفیت اعضای ensemble وابسته است."
    )
    consensus_display = consensus_selection.copy()
    consensus_display["silhouette"] = consensus_display["silhouette"].map(lambda value: f"{value:.4f}")
    consensus_display["minimum_cluster_fraction"] = consensus_display["minimum_cluster_fraction"].map(_pct)
    report.table(
        ["k", "Silhouette", "کمترین سهم خوشه", "تعداد خوشه"],
        consensus_display[["k", "silhouette", "minimum_cluster_fraction", "clusters"]].itertuples(index=False, name=None),
    )
    report.callout(
        "انتخاب نهایی",
        f"k={phase3['consensus_k']} با silhouette={selected_consensus['silhouette']:.4f} و رعایت حداقل سهم ۲٪ انتخاب شد. "
        "این انتخاب یک خلاصه دوپروفایلی است و اختلاف شواهد k در فصل قبل عمدا در تفسیر حفظ می‌شود.",
    )
    report.figure(figures / "phase3_consensus_selection.png", "شکل ۱۲ — انتخاب k در فضای consensus")
    report.heading("۷.۲. تعمیم به کل cohort", 2)
    report.paragraph(
        f"برچسب consensus مستقیما برای نمونه {p2_families['sample_size']:,}تایی تعریف شد. برای انتساب کل {p1_cleaning['rows_cleaned']:,} "
        f"ردیف، یک Random Forest متوازن روی ۵۰ مؤلفه PCA آموزش و با split طبقه‌بندی‌شده ارزیابی شد. دقت holdout برابر "
        f"{_pct(phase3['validation_accuracy'])} بود. centroidهای consensus نیز برای انتساب زنده و محاسبه فاصله هر رکورد تا تمام خوشه‌ها ذخیره شدند."
    )

    report.heading("۸. تفسیر خوشه‌ها و توضیح‌پذیری", 1)
    report.heading("۸.۱. پروفایل‌های نهایی", 2)
    report.table(
        ["خوشه", "نام پروفایل", "تعداد", "سهم", "میانه تجربه", "میانه breadth", "نقش غالب"],
        [
            (
                row.cluster,
                row.profile_name,
                f"{row.respondents:,}",
                f"{row.share_pct:.1f}٪",
                row.ExperienceConsensus_median,
                row.TechnologyBreadth_median,
                row.DevType_mode,
            )
            for row in profiles.itertuples()
        ],
    )
    report.paragraph(
        "خوشه ۰ «کارورزان با پشته متمرکزتر» نام‌گذاری شد: سهم ۷۶٫۶٪ و میانه ۱۴ فناوری. خوشه ۱ «توسعه‌دهندگان "
        "چندفناوری گسترده» است: سهم ۲۳٫۴٪ و میانه ۳۲ فناوری. مرز اصلی یک طیف breadth است، نه تمایز ارزشی میان افراد. "
        "تفاوت نقش، تحصیلات، کشور و پذیرش AI برای غنی‌کردن تفسیر گزارش می‌شود، اما علت عضویت محسوب نمی‌شود."
    )
    report.figure(figures / "phase3_cluster_sizes.png", "شکل ۱۳ — اندازه و سهم دو پروفایل consensus")
    report.heading("۸.۲. فناوری‌های شاخص", 2)
    top_technologies = (
        technologies.sort_values(["cluster", "pct_cluster"], ascending=[True, False])
        .groupby("cluster", as_index=False)
        .head(10)
    )
    report.table(
        ["خوشه", "حوزه", "فناوری", "پاسخ‌دهنده", "درصد خوشه"],
        [
            (row.cluster, row.domain, row.technology, f"{row.respondents:,}", f"{row.pct_cluster:.1f}٪")
            for row in top_technologies.itertuples()
        ],
    )
    report.heading("۸.۳. نمونه‌ها، قواعد و SHAP", 2)
    report.paragraph(
        "برای هر خوشه سه نمونه ذخیره شد: medoid با کمترین مجموع فاصله درون‌خوشه‌ای، exemplar با بیشترین silhouette و boundary "
        "با کمترین silhouette. این نقاط امکان می‌دهند معنای مرکز، نمونه بسیار نماینده و ناحیه تردید هر پروفایل جداگانه بررسی شود."
    )
    report.table(
        ["خوشه", "نوع نمونه", "ResponseId", "Silhouette"],
        [
            (row.cluster, row.kind, int(row.ResponseId), f"{row.silhouette:.4f}")
            for row in exemplars.itertuples()
        ],
    )
    report.paragraph(
        f"درخت تصمیم عمق چهار عضویت را با fidelity برابر {_pct(phase3['explanation_tree_validation_accuracy'])} روی holdout و "
        f"{_pct(phase3['explanation_tree_training_accuracy'])} روی fit نهایی بازسازی کرد. SHAP نیز با وضعیت {phase3['shap_status']} "
        "اجرا شد. چون ویژگی‌های ورودی درخت مؤلفه‌های PCA هستند، تفسیر SHAP باید همراه با loadings، پروفایل و فناوری‌های شاخص انجام شود."
    )
    top_shap = shap_values.sort_values("shap_mean_abs", ascending=False).head(12)
    report.table(
        ["ویژگی", "اهمیت مدل", "میانگین |SHAP|"],
        [
            (row.feature, f"{row.importance:.4f}", f"{row.shap_mean_abs:.4f}")
            for row in top_shap.itertuples()
        ],
    )
    report.figure(figures / "phase3_shap_importance.png", "شکل ۱۴ — اهمیت SHAP مؤلفه‌های مؤثر بر انتساب")

    report.heading("۹. تحلیل پایین‌دستی و ناهنجاری", 1)
    report.heading("۹.۱. مدل جبران خدمت", 2)
    report.paragraph(
        "برای سنجش ارزش عملی خوشه‌ها، Random Forest روی log جبران خدمت یک‌بار به‌صورت سراسری و یک‌بار به‌صورت شرطی بر خوشه "
        "برازش شد. اگر خوشه‌ها ساختار پیش‌بینی‌پذیر مستقلی حمل می‌کردند، مدل‌های شرطی باید MAE کمتری می‌داشتند. این بهبود در هیچ خوشه‌ای رخ نداد."
    )
    report.table(
        ["مدل", "خوشه", "ردیف آزمون", "MAE log", "MAE مدل سراسری روی همان ردیف"],
        [
            (row.model, row.cluster, row.test_rows, f"{row.mae_log:.4f}", f"{row.global_mae_same_rows:.4f}")
            for row in downstream.itertuples()
        ],
    )
    report.callout(
        "نتیجه تحلیل جبران خدمت",
        "تقسیم consensus دقت پیش‌بینی جبران خدمت را بهتر نکرد. بنابراین نتایج این بخش، کاربرد پیش‌بینی مستقلی برای خوشه‌ها نشان نمی‌دهد.",
    )
    report.heading("۹.۲. تشخیص ناهنجاری خوشه‌محور", 2)
    report.paragraph(
        "Isolation Forest با نرخ آلودگی یک درصد جداگانه در هر خوشه برازش شد و سپس یک درصد بالای امتیازها در کل cohort برای بازبینی ذخیره شد. ناهنجاری در اینجا "
        "به معنای فاصله از الگوی رایج همان خوشه است، نه خطا یا رفتار نامطلوب. این تعریف خوشه‌محور مانع از آن می‌شود که اعضای یک خوشه کوچک صرفا به دلیل تفاوت کلی ناهنجار تلقی شوند."
    )
    anomaly_preview = anomalies.sort_values(["cluster", "anomaly_score"], ascending=[True, False]).groupby("cluster").head(5)
    report.table(
        ["خوشه", "ResponseId", "امتیاز ناهنجاری"],
        [
            (row.cluster, int(row.ResponseId), f"{row.anomaly_score:.4f}")
            for row in anomaly_preview.itertuples()
        ],
    )

    report.heading("۱۰. انصاف، composition و حساسیت", 1)
    report.heading("۱۰.۱. ممیزی composition", 2)
    dominant_composition = (
        composition.sort_values("share", ascending=False)
        .groupby(["attribute", "cluster"], as_index=False)
        .head(1)
    )
    report.table(
        ["ویژگی", "خوشه", "دسته غالب", "سهم"],
        [
            (row.attribute, row.cluster, row.category, _pct(row.share))
            for row in dominant_composition.itertuples()
        ],
    )
    report.paragraph(
        "تفاوت composition باید در تفسیر و استفاده احتمالی از مدل دیده شود. بااین‌حال این جدول فقط توصیف همبستگی است؛ "
        "متغیرهای بررسی‌شده لزوما sensitive attribute کامل یا علت عضویت نیستند. هیچ تصمیم فردی نباید صرفا بر اساس شناسه خوشه گرفته شود."
    )
    report.heading("۱۰.۲. حساسیت به پیش‌پردازش", 2)
    sensitivity_display = sensitivity.sort_values("ari_vs_consensus", ascending=False).copy()
    sensitivity_display["ari_vs_consensus"] = sensitivity_display["ari_vs_consensus"].map(lambda value: f"{value:.3f}")
    report.table(
        ["گونه آزمایش", "ARI نسبت به consensus"],
        sensitivity_display[["variant", "ari_vs_consensus"]].itertuples(index=False, name=None),
    )
    report.paragraph(
        "دامنه ARI از حدود ۰٫۲۶ تا ۰٫۹۴ نشان می‌دهد نمایش و scaler بخشی از پاسخ clustering هستند. PCA-10 بیشترین توافق را داشت، "
        "در حالی که نمایش کامل Standard-scaled کمترین توافق را ایجاد کرد. این حساسیت ادعای «ذاتی‌بودن» دو خوشه را تضعیف می‌کند و "
        "ضرورت گزارش سناریوهای جایگزین را نشان می‌دهد."
    )
    report.figure(figures / "phase3_sensitivity.png", "شکل ۱۵ — حساسیت برچسب نهایی به نمایش، scaler و کاهش بعد")

    report.heading("۱۱. مسیر پیشرفته دوم و امتیازهای افزوده", 1)
    report.heading("۱۱.۱. NMF پربعد", 2)
    report.paragraph(
        f"NMF مستقیما روی {p1_features['technology_features']} شاخص نامنفی فناوری اجرا شد تا ساختار اکوسیستم فناوری، مستقل از "
        "تقسیم breadth، بررسی شود. از میان ۲ تا ۶ مؤلفه، راه‌حل چهارمؤلفه‌ای با silhouette کسینوسی "
        f"{bonus['nmf']['technology_cosine_silhouette']:.3f} انتخاب شد. ARI آن با consensus برابر {bonus['nmf']['ari_vs_consensus']:.3f} "
        "و NMI برابر ۰٫۰۱۹ است؛ بنابراین دو مسیر ساختارهای مکمل می‌یابند."
    )
    report.table(
        ["مؤلفه", "Silhouette کسینوسی", "Silhouette نهفته", "کمترین سهم", "ARI", "انتخاب"],
        [
            (
                row.components,
                f"{row.technology_cosine_silhouette:.3f}",
                f"{row.latent_silhouette:.3f}",
                _pct(row.minimum_cluster_fraction),
                f"{row.ari_vs_consensus:.3f}",
                "بله" if row.selected else "خیر",
            )
            for row in nmf.itertuples()
        ],
    )
    report.figure(figures / "bonus_nmf_comparison.png", "شکل ۱۶ — مقایسه تعداد مؤلفه‌ها در مسیر NMF")
    report.heading("۱۱.۲. فاصله اطمینان و permutation", 2)
    silhouette_ci = confidence.loc[confidence["metric"] == "silhouette"].copy()
    report.table(
        ["روش", "برآورد", "کران پایین ۹۵٪", "کران بالا ۹۵٪"],
        [
            (row.method, f"{row.estimate:.4f}", f"{row.ci_lower:.4f}", f"{row.ci_upper:.4f}")
            for row in silhouette_ci.itertuples()
        ],
    )
    report.paragraph(
        f"با {bonus['significance']['bootstrap_repeats']} بازنمونه‌گیری برای پنج معیار و "
        f"{bonus['significance']['permutation_repeats']:,} جایگشت جفتی، روش {bonus['significance']['best_method']} نسبت به "
        f"{bonus['significance']['runner_up']} اختلاف silhouette برابر {bonus['significance']['mean_silhouette_difference']:.4f} "
        f"با p={bonus['significance']['permutation_pvalue']:.4g} داشت. معناداری آماری به‌تنهایی به معنای اهمیت عملی بزرگ نیست؛ اندازه اثر نیز گزارش شده است."
    )
    report.figure(figures / "bonus_silhouette_confidence_intervals.png", "شکل ۱۷ — فاصله اطمینان bootstrap برای silhouette روش‌های نامزد")
    report.heading("۱۱.۳. پایداری split و UMAP سه‌بعدی", 2)
    report.paragraph(
        f"داده در میانه ExperienceConsensus={bonus['split_stability']['threshold']:.0f} به دو زیرجمعیت تقسیم و هر نیمه مستقل "
        f"خوشه‌بندی شد. تطبیق prototype با الگوریتم Hungarian میانگین Jaccard فناوری {bonus['split_stability']['mean_top20_technology_jaccard']:.3f} "
        "را نشان داد؛ پایداری متوسط است، نه یکسانی کامل. UMAP سه‌بعدی تعاملی نیز برای "
        f"{bonus['umap_3d']['sample_size']:,} پاسخ ساخته شد و در داشبورد و فایل HTML مستقل قرار دارد."
    )
    report.table(
        ["خوشه split اول", "خوشه split دوم", "ردیف اول", "ردیف دوم", "Jaccard فناوری"],
        [
            (row.cluster_a, row.cluster_b, row.rows_a, row.rows_b, f"{row.top20_technology_jaccard:.3f}")
            for row in split.itertuples()
        ],
    )
    report.figure(figures / "bonus_split_stability.png", "شکل ۱۸ — پایداری prototype میان دو نیمه تجربه")

    report.heading("۱۲. خط تولید، بازتولیدپذیری و داشبورد", 1)
    report.heading("۱۲.۱. معماری pipeline", 2)
    report.paragraph(
        "خط تولید به مراحل ingest، clean، feature، scale، reduce، cluster و evaluate تفکیک شده است. خروجی هر مرحله در data/interim، "
        "data/processed یا artifacts ذخیره می‌شود و مرحله‌های پایین‌دستی می‌توانند بدون تکرار کل محاسبات اجرا شوند. schema ورودی کنترل، "
        "مدل‌ها با joblib سریال و manifestهای ورودی و خروجی با SHA-256 نسخه‌بندی شده‌اند."
    )
    report.table(
        ["لایه", "خروجی اصلی", "کاربرد"],
        [
            ["ورود و پاک‌سازی", "Parquet میانی + manifest", "ردیابی منبع و تصمیم‌های کیفیت"],
            ["ویژگی", "ماتریس‌های sparse + metadata", "نمایش کامل و فناوری"],
            ["کاهش بعد", "PCA/SVD/UMAP + مدل", "نمایش محاسباتی و بصری"],
            ["خوشه‌بندی", "labels، centroid، co-association", "تحلیل و انتساب"],
            ["ارزیابی", "CSV/JSON + شکل", "گزارش و audit"],
            ["Registry", "مدل، هش، پارامتر و scoreboard", "استقرار و بازبینی"],
        ],
    )
    report.heading("۱۲.۲. پایش drift", 2)
    drift_preview = drift.sort_values(["alert", "psi"], ascending=[False, False]).head(10).copy()
    report.table(
        ["ویژگی", "PSI", "KS", "p-value", "هشدار"],
        [
            (row.feature, f"{row.psi:.4f}", f"{row.ks_statistic:.4f}", f"{row.ks_pvalue:.4g}", row.alert)
            for row in drift_preview.itertuples()
        ],
    )
    report.paragraph(
        "baseline drift با دو نیمه ثابت داده برای ده مؤلفه نخست ساخته شد. PSI بزرگ‌تر از ۰٫۲ یا KS بزرگ‌تر از ۰٫۱ آستانه هشدار "
        "عملیاتی است. این baseline جایگزین داده زمانی واقعی نیست، اما قرارداد فنی لازم برای پایش نسخه‌های آینده را فراهم می‌کند."
    )
    report.heading("۱۲.۳. داشبورد تعاملی", 2)
    report.paragraph(
        "داشبورد Streamlit پنج صفحه دارد: Overview، Cluster Explorer با UMAP دوبعدی و فیلتر تجربه/خوشه، Evaluation با معیارها و "
        "پایداری، Live Assignment برای رکورد یا CSV با فاصله تا تمام centroidها، و 3D Explorer برای UMAP تعاملی ده‌هزارنقطه‌ای. "
        "برچسب‌گذاری زنده فقط رکوردهای دارای PC1 تا PC50 را می‌پذیرد تا قرارداد پیش‌پردازش شفاف بماند."
    )
    report.heading("۱۲.۴. دستور بازتولید", 2)
    report.code_block("python -m venv venv")
    report.code_block(".\\venv\\Scripts\\python.exe -m pip install -r requirements.txt")
    report.code_block(".\\venv\\Scripts\\python.exe -m scripts.run_all --config config.yaml")
    report.code_block(".\\venv\\Scripts\\python.exe -m scripts.build_all_deliverables")
    report.code_block("powershell -ExecutionPolicy Bypass -File scripts\\export_reports_pdf.ps1")
    report.code_block(".\\venv\\Scripts\\python.exe -m scripts.validate_release")
    report.code_block(".\\venv\\Scripts\\python.exe -m pytest -q")

    report.heading("۱۳. بحث و محدودیت‌ها", 1)
    report.bullet("اختلاف Silhouette، Kneedle، Gap، BIC و bootstrap نشان می‌دهد k واحد و بدیهی وجود ندارد.")
    report.bullet("Silhouette نهایی متوسط است و UMAP/VAT هم‌پوشانی پیوسته را نشان می‌دهند؛ مرزها قطعی نیستند.")
    report.bullet("پاسخ به scaler و تعداد مؤلفه‌های PCA حساس است؛ بخشی از ساختار حاصل انتخاب نمایش است.")
    report.bullet("پیمایش خوداظهاری، داوطلبانه و مقطعی است؛ نتیجه به کل جمعیت توسعه‌دهندگان یا سال‌های دیگر تعمیم قطعی ندارد.")
    report.bullet("متغیرهای نقش، تحصیلات و AI برچسب مرجع نیستند و NMI پایین آن‌ها شکست مدل محسوب نمی‌شود.")
    report.bullet("مدل‌های جبران خدمت شرطی بر خوشه بهتر نشدند؛ ارزش پایین‌دستی segmentation محدود است.")
    report.bullet("تفسیر SHAP روی PCA غیرمستقیم است و باید همراه loadings و پروفایل فناوری خوانده شود.")
    report.bullet("اعتبارسنجی انسانی و داده سال‌های بعد می‌تواند پایداری معنایی و زمانی پروفایل‌ها را بهتر ارزیابی کند.")
    report.callout(
        "محدوده کاربرد نتیجه",
        "خوشه‌ها برای اکتشاف، خلاصه‌سازی و تولید فرضیه مناسب‌اند. استفاده از آن‌ها برای قضاوت درباره صلاحیت، ارزش یا آینده یک فرد "
        "از داده و طراحی این مطالعه پشتیبانی نمی‌شود.",
    )

    report.heading("۱۴. نتیجه‌گیری", 1)
    report.paragraph(
        "نتایج نشان داد داده Stack Overflow 2024 ساختار غیرتصادفی دارد، اما این ساختار چندمقیاسی و هم‌پوشان است. بررسی پنج خانواده "
        "الگوریتم، چند معیار k، پایداری و consensus به یک تقسیم دوپروفایلی رسید که عمدتا breadth پشته فناوری را خلاصه می‌کند. "
        "پروفایل متمرکزتر اکثریت cohort و پروفایل چندفناوری گسترده اقلیت بزرگ آن را تشکیل می‌دهد. دقت بالای assigner و درخت توضیحی "
        "نشان می‌دهد این تقسیم از نظر محاسباتی قابل بازتولید و توضیح است، اما silhouette متوسط، حساسیت پیش‌پردازش و نتیجه منفی مدل "
        "جبران خدمت مانع از ادعاهای قوی‌تر می‌شوند."
    )
    report.paragraph(
        "در کنار برچسب نهایی، فرایند کامل اجرا، خروجی‌های نسخه‌دار، مقایسه روش‌ها، تحلیل خطا و حساسیت، مسیر مکمل NMF، "
        "آزمون‌های آماری و داشبورد نیز تهیه شد. برای ادامه کار می‌توان همین فرایند را روی پیمایش سال‌های دیگر اجرا و تغییر "
        "توزیع داده و ثبات معنایی پروفایل‌ها را بررسی کرد."
    )

    report.heading("منابع", 1)
    report.paragraph("[۱] Stack Overflow. Developer Survey 2024. https://survey.stackoverflow.co/2024/")
    report.paragraph("[۲] Pedregosa, F. et al. Scikit-learn: Machine Learning in Python. JMLR, 2011.")
    report.paragraph("[۳] Rousseeuw, P. J. Silhouettes: A Graphical Aid to Cluster Analysis. JCAM, 1987.")
    report.paragraph("[۴] Ester, M. et al. A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases. KDD, 1996.")
    report.paragraph("[۵] McInnes, L., Healy, J., Melville, J. UMAP: Uniform Manifold Approximation and Projection, 2018.")
    report.paragraph("[۶] Monti, S. et al. Consensus Clustering: A Resampling-Based Method for Class Discovery. Machine Learning, 2003.")
    report.paragraph("[۷] Lee, D. D., Seung, H. S. Learning the Parts of Objects by Non-negative Matrix Factorization. Nature, 1999.")
    report.paragraph("[۸] Lundberg, S. M., Lee, S.-I. A Unified Approach to Interpreting Model Predictions. NeurIPS, 2017.")

    report.heading("پیوست الف — شبکه پارامترها", 1)
    report.table(
        ["بخش", "پارامتر", "مقدار"],
        [
            ["عمومی", "random_state", config["project"]["random_state"]],
            ["ورود", "chunk_size", config["data"]["chunk_size"]],
            ["ویژگی", "scaler اصلی", config["features"]["selected_scaler"]],
            ["کاهش بعد", "PCA components", config["features"]["pca_components"]],
            ["UMAP", "neighbors / min_dist", f"{config['features']['umap_neighbors']} / {config['features']['umap_min_dist']}"],
            ["Phase 2", "k values", ", ".join(map(str, config["phase2"]["k_values"]))],
            ["پایداری", "seed / bootstrap", f"{config['phase2']['stability_seeds']} / {config['phase2']['bootstrap_repeats']}"],
            ["مقایسه خانواده", "sample / minimum share", f"{config['phase2']['family_sample_size']:,} / {_pct(config['phase2']['minimum_cluster_fraction'])}"],
            ["NMF", "components", ", ".join(map(str, config["bonus"]["nmf_components"]))],
            ["Permutation", "repeats", f"{config['bonus']['permutation_repeats']:,}"],
        ],
    )

    report.heading("پیوست ب — جدول کامل راه‌حل‌های برتر فاز دوم", 1)
    appendix_rows = (
        family_scores.dropna(subset=["silhouette"])
        .sort_values("silhouette", ascending=False)
        .head(20)
    )
    report.table(
        ["خانواده", "الگوریتم", "k درخواستی", "خوشه یافت‌شده", "نویز", "Silhouette", "DB", "CH"],
        [
            (
                row.family,
                row.algorithm,
                row.k_requested,
                row.clusters_found,
                _pct(row.noise_fraction),
                f"{row.silhouette:.4f}",
                f"{row.davies_bouldin:.4f}",
                f"{row.calinski_harabasz:.1f}",
            )
            for row in appendix_rows.itertuples()
        ],
    )

    output = ROOT / "reports" / "Final_Report_FA.docx"
    report.save(output)
    return output


if __name__ == "__main__":
    print(build())
