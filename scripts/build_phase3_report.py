from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stackoverflow_clustering.config import load_config
from stackoverflow_clustering.reporting import PersianReport, add_project_cover


def build() -> Path:
    config = load_config(ROOT / "config.yaml")
    artifacts = ROOT / "artifacts"
    tables = ROOT / "reports" / "tables"
    figures = ROOT / "reports" / "figures"

    summary = json.loads((artifacts / "phase3_run_summary.json").read_text(encoding="utf-8"))
    bonus = json.loads((artifacts / "bonus_summary.json").read_text(encoding="utf-8"))
    consensus = pd.read_csv(tables / "phase3_consensus_k_selection.csv")
    profiles = pd.read_csv(tables / "phase3_cluster_profiles.csv")
    technologies = pd.read_csv(tables / "phase3_cluster_top_technologies.csv")
    exemplars = pd.read_csv(tables / "phase3_exemplars_boundaries.csv")
    shap_values = pd.read_csv(tables / "phase3_feature_importance_shap.csv")
    downstream = pd.read_csv(tables / "phase3_downstream_compensation.csv")
    anomalies = pd.read_csv(tables / "phase3_cluster_anomalies.csv")
    composition = pd.read_csv(tables / "phase3_composition_audit.csv")
    sensitivity = pd.read_csv(tables / "phase3_sensitivity.csv")
    drift = pd.read_csv(tables / "phase3_drift_baseline.csv")
    nmf = pd.read_csv(tables / "bonus_nmf_scores.csv")
    confidence = pd.read_csv(tables / "bonus_metric_confidence_intervals.csv")
    permutation = pd.read_csv(tables / "bonus_permutation_tests.csv")
    split = pd.read_csv(tables / "bonus_split_stability.csv")

    report = PersianReport("Consensus، پروفایل‌سازی، توضیح‌پذیری و تحلیل پیشرفته")
    add_project_cover(report, "گزارش فاز سوم پروژه داده‌کاوی پیشرفته")
    report.add_toc()

    report.heading("چکیده اجرایی", 1)
    report.paragraph(
        "در فاز سوم چهار نامزد مکمل K-Means، Ward، GMM-full-k3 و Spectral-k3 در یک ماتریس co-association تجمیع شدند. "
        f"برش Agglomerative در فضای consensus مقدار k={summary['consensus_k']} را انتخاب کرد. برچسب نمونه مشترک با یک "
        f"assigner متوازن به تمام ۶۰٬۰۲۳ پاسخ تعمیم یافت و دقت holdout آن {100 * summary['validation_accuracy']:.1f}٪ بود. "
        f"درخت توضیحی نیز fidelity برابر {100 * summary['explanation_tree_validation_accuracy']:.1f}٪ به دست آورد."
    )
    report.paragraph(
        f"دو پروفایل نهایی {summary['cluster_sizes']['0']:,} و {summary['cluster_sizes']['1']:,} عضو دارند و تفاوت اصلی آن‌ها "
        "پهنای پشته فناوری با میانه‌های ۱۴ و ۳۲ فناوری است. مدل‌های جبران خدمت شرطی بر خوشه نسبت به مدل سراسری بهتر نشدند و "
        "تحلیل حساسیت، وابستگی قابل‌توجه برچسب‌ها به scaler و نمایش را نشان داد. مسیر دوم NMF چهار اکوسیستم فناوری با ARI نزدیک صفر "
        "نسبت به consensus یافت؛ بنابراین ساختار فناوری‌محور و تقسیم breadth مکمل‌اند."
    )
    report.callout(
        "نتیجه تفسیری",
        "Consensus یک خلاصه قابل‌توضیح از طیف breadth فناوری است، نه کشف دو نوع ذاتی توسعه‌دهنده. هم‌پوشانی، حساسیت و "
        "نتیجه تحلیل پایین‌دستی نیز باید هنگام ارائه یا استفاده از خوشه‌ها در نظر گرفته شود.",
    )
    report.table(
        ["شاخص", "مقدار"],
        [
            ["نامزدهای consensus", "K-Means، Ward، GMM، Spectral"],
            ["k نهایی", summary["consensus_k"]],
            ["اندازه خوشه‌ها", f"{summary['cluster_sizes']['0']:,} / {summary['cluster_sizes']['1']:,}"],
            ["دقت assigner", f"{100 * summary['validation_accuracy']:.1f}٪"],
            ["Fidelity درخت", f"{100 * summary['explanation_tree_validation_accuracy']:.1f}٪"],
            ["مسیر دوم", f"NMF با {bonus['nmf']['selected_components']} مؤلفه"],
        ],
    )

    report.heading("۱. ساخت Consensus Clustering", 1)
    report.heading("۱.۱. منطق ensemble", 2)
    report.paragraph(
        "نتایج فاز دوم نشان دادند معیارهای k و خانواده‌های الگوریتم توافق کامل ندارند. به‌جای انتخاب یک مدل منفرد، برای هر زوج "
        "پاسخ‌دهنده نسبت الگوریتم‌هایی که آن دو را هم‌خوشه می‌دانند محاسبه شد. این co-association نتیجه الگوریتم‌هایی با فرض‌های متفاوت را "
        "جمع می‌کند و وابستگی به هندسه کروی، سلسله‌مراتبی، احتمالاتی یا گرافی منفرد را کاهش می‌دهد."
    )
    report.paragraph(
        "فاصله consensus برابر یک منهای co-association تعریف و Agglomerative با average linkage برای kهای ۲ تا ۶ اجرا شد. "
        "همان قاعده حداقل سهم ۲٪ حفظ شد تا پاسخ‌های منحط انتخاب نشوند. Silhouette در فضای PCA مشترک برای مقایسه برش‌ها استفاده شد."
    )
    consensus_display = consensus.copy()
    consensus_display["silhouette"] = consensus_display["silhouette"].map(lambda value: f"{value:.4f}")
    consensus_display["minimum_cluster_fraction"] = consensus_display["minimum_cluster_fraction"].map(
        lambda value: f"{100 * value:.1f}٪"
    )
    report.table(
        ["k", "Silhouette", "کمترین سهم", "خوشه واقعی"],
        consensus_display[["k", "silhouette", "minimum_cluster_fraction", "clusters"]].itertuples(index=False, name=None),
    )
    report.figure(figures / "phase3_consensus_selection.png", "شکل ۱ — انتخاب تعداد خوشه در فضای consensus")

    report.heading("۲. تعمیم برچسب و مدل انتساب", 1)
    report.paragraph(
        "ماتریس co-association فقط روی نمونه مشترک ۳۰۰۰تایی قابل محاسبه بود. برای تعمیم، Random Forest متوازن روی ۵۰ مؤلفه PCA "
        "آموزش دید و با split طبقه‌بندی‌شده ارزیابی شد. دقت ۹۶٫۴٪ نشان می‌دهد مرز consensus در فضای PCA قابل تقریب است، اما "
        "assigner جایگزین تعریف اصلی consensus نیست و فقط قرارداد عملیاتی انتساب را فراهم می‌کند."
    )
    report.paragraph(
        "پس از ارزیابی، مدل روی کل نمونه consensus بازبرازش و برای تمام cohort پیش‌بینی شد. centroid هر خوشه نیز ذخیره شد تا داشبورد "
        "برای رکورد جدید علاوه بر برچسب، فاصله تا تمام centroidها را نمایش دهد. این فاصله شاخص عدم‌قطعیت نسبی است و probability کالیبره‌شده نیست."
    )
    report.callout(
        "کنترل تعمیم",
        f"دقت holdout assigner برابر {100 * summary['validation_accuracy']:.1f}٪ است. این عدد روی داده نگه‌داشته‌شده محاسبه "
        "شده و از دقت آموزش درخت توضیحی جدا نگه داشته شده است.",
    )

    report.heading("۳. پروفایل‌های انسانی", 1)
    report.table(
        ["خوشه", "پروفایل", "تعداد", "سهم", "تجربه میانه", "Breadth میانه", "نقش غالب"],
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
        "خوشه ۰ با ۷۶٫۶٪ cohort و میانه ۱۴ فناوری «پشته متمرکزتر» نام گرفت. خوشه ۱ با ۲۳٫۴٪ و میانه ۳۲ فناوری "
        "«چندفناوری گسترده» است. این نام‌ها بر ویژگی غالب breadth متکی‌اند. تفاوت تجربه و نقش به تفسیر کمک می‌کند، اما نام‌گذاری "
        "نباید به سلسله‌مراتب ارزشی میان توسعه‌دهندگان تبدیل شود."
    )
    report.figure(figures / "phase3_cluster_sizes.png", "شکل ۲ — اندازه و سهم دو پروفایل consensus")

    report.heading("۴. فناوری‌های متمایزکننده", 1)
    report.paragraph(
        "برای هر یک از هفت حوزه فناوری، فراوانی گزینه‌ها در هر خوشه محاسبه و هشت فناوری شاخص ذخیره شد. فناوری‌های عمومی مانند "
        "JavaScript، HTML/CSS، SQL و Python در هر دو گروه دیده می‌شوند؛ تفاوت اصلی در تعداد و هم‌وقوعی فناوری‌ها است. "
        "پروفایل گسترده‌تر سهم بالاتری از ابزارها، پلتفرم‌ها و چارچوب‌های هم‌زمان دارد."
    )
    top_technologies = (
        technologies.sort_values(["cluster", "pct_cluster"], ascending=[True, False])
        .groupby("cluster")
        .head(12)
    )
    report.table(
        ["خوشه", "حوزه", "فناوری", "پاسخ‌دهنده", "درصد خوشه"],
        [
            (row.cluster, row.domain, row.technology, f"{row.respondents:,}", f"{row.pct_cluster:.1f}٪")
            for row in top_technologies.itertuples()
        ],
    )

    report.heading("۵. Medoid، exemplar و boundary", 1)
    report.paragraph(
        "Medoid دقیق با کمینه‌کردن مجموع فاصله‌های درون‌خوشه‌ای در نمونه consensus تعیین شد. exemplar بیشترین silhouette و "
        "boundary کمترین silhouette را دارد. medoid تصویر مرکز واقعی، exemplar نمونه بسیار نماینده و boundary ناحیه ابهام بین "
        "پروفایل‌ها را نشان می‌دهد. ResponseId برای بازبینی این نمونه‌ها در notebook ذخیره شده است."
    )
    report.table(
        ["خوشه", "نوع", "شاخص ردیف", "ResponseId", "Silhouette"],
        [
            (row.cluster, row.kind, row.row_index, int(row.ResponseId), f"{row.silhouette:.4f}")
            for row in exemplars.itertuples()
        ],
    )

    report.heading("۶. قواعد توضیحی و SHAP", 1)
    report.paragraph(
        f"یک درخت تصمیم با عمق چهار روی برچسب consensus آموزش داده شد. fidelity روی holdout برابر "
        f"{100 * summary['explanation_tree_validation_accuracy']:.1f}٪ و fit نهایی روی کل داده "
        f"{100 * summary['explanation_tree_training_accuracy']:.1f}٪ بود. قواعد کامل در artifact متنی ذخیره شدند."
    )
    report.paragraph(
        "SHAP اهمیت و جهت اثر مؤلفه‌ها را برای classifier توضیحی برآورد کرد. چون PCها ترکیب خطی ویژگی‌های اصلی‌اند، یک PC به‌تنهایی "
        "معادل مفهوم دامنه‌ای نیست؛ نتیجه SHAP باید همراه پروفایل‌های فناوری، loadings و نمونه‌های واقعی خوانده شود."
    )
    top_shap = shap_values.sort_values("shap_mean_abs", ascending=False).head(15)
    report.table(
        ["مؤلفه", "اهمیت مدل", "میانگین |SHAP|"],
        [
            (row.feature, f"{row.importance:.4f}", f"{row.shap_mean_abs:.4f}")
            for row in top_shap.itertuples()
        ],
    )
    report.figure(figures / "phase3_shap_importance.png", "شکل ۳ — اهمیت SHAP مؤلفه‌های مؤثر بر انتساب")

    report.heading("۷. تحلیل پایین‌دستی جبران خدمت", 1)
    report.paragraph(
        "Random Forest روی log جبران خدمت یک‌بار سراسری و سپس جداگانه در هر خوشه برازش شد. معیار MAE روی ردیف‌های آزمون مشترک "
        "مقایسه شد. مدل‌های شرطی در هیچ خوشه‌ای از مدل سراسری بهتر نبودند؛ بنابراین تقسیم‌بندی خوشه‌ای برای این پیامد دقت "
        "پیش‌بینی را بهتر نکرد."
    )
    report.table(
        ["مدل", "خوشه", "ردیف آزمون", "MAE log", "MAE سراسری"],
        [
            (row.model, row.cluster, row.test_rows, f"{row.mae_log:.4f}", f"{row.global_mae_same_rows:.4f}")
            for row in downstream.itertuples()
        ],
    )
    report.callout(
        "نتیجه تحلیل جبران خدمت",
        "خوشه‌ها جبران خدمت را بهتر پیش‌بینی نکردند؛ بنابراین نباید از آن‌ها به‌عنوان شاخص ارزش اقتصادی یا سطح حرفه‌ای استفاده شود.",
    )

    report.heading("۸. ناهنجاری خوشه‌محور", 1)
    report.paragraph(
        "Isolation Forest با نرخ آلودگی یک درصد در هر خوشه جداگانه برازش و سپس یک درصد بالای امتیازها در کل cohort برای بازبینی انتخاب شد. ناهنجاری به معنای فاصله از الگوی "
        "رایج همان خوشه است، نه داده بد یا فرد نامطلوب. برازش درون‌خوشه‌ای مانع از آن می‌شود که اعضای یک پروفایل کوچک فقط به دلیل "
        "تفاوت با خوشه اکثریت ناهنجار محسوب شوند."
    )
    anomaly_preview = anomalies.sort_values(["cluster", "anomaly_score"], ascending=[True, False]).groupby("cluster").head(8)
    report.table(
        ["خوشه", "ResponseId", "امتیاز ناهنجاری"],
        [
            (row.cluster, int(row.ResponseId), f"{row.anomaly_score:.4f}")
            for row in anomaly_preview.itertuples()
        ],
    )

    report.heading("۹. ممیزی composition و انصاف", 1)
    report.paragraph(
        "ترکیب تحصیلات، دورکاری، کشور و پذیرش AI در هر خوشه محاسبه شد. هدف آشکارکردن stratification احتمالی است، نه اثبات رابطه "
        "علی. دسته‌های غالب نشان می‌دهند خوشه‌ها چگونه با metadata هم‌زمان تغییر می‌کنند و کجا استفاده عملی ممکن است متغیرهای جانشین "
        "نامطلوب ایجاد کند."
    )
    dominant = composition.sort_values("share", ascending=False).groupby(["attribute", "cluster"]).head(1)
    report.table(
        ["ویژگی", "خوشه", "دسته غالب", "سهم"],
        [
            (row.attribute, row.cluster, row.category, f"{100 * row.share:.1f}٪")
            for row in dominant.itertuples()
        ],
    )
    report.callout(
        "حد اخلاقی",
        "هیچ تصمیم فردی، استخدامی یا رتبه‌بندی نباید صرفا بر اساس خوشه انجام شود. composition فقط برای audit و تفسیر گروهی گزارش می‌شود.",
    )

    report.heading("۱۰. حساسیت و استحکام", 1)
    report.paragraph(
        "انتساب نهایی با PCA-10، PCA-20، PCA-50، نمایش کامل و سه scaler مقایسه شد. ARI نسبت به consensus میزان حفظ ساختار را "
        "می‌سنجد. دامنه وسیع ARI نشان می‌دهد بخش مهمی از پاسخ clustering به انتخاب هندسه پیش‌پردازش وابسته است."
    )
    sensitivity_display = sensitivity.sort_values("ari_vs_consensus", ascending=False).copy()
    sensitivity_display["ari_vs_consensus"] = sensitivity_display["ari_vs_consensus"].map(lambda value: f"{value:.3f}")
    report.table(
        ["سناریو", "ARI نسبت به consensus"],
        sensitivity_display[["variant", "ari_vs_consensus"]].itertuples(index=False, name=None),
    )
    report.figure(figures / "phase3_sensitivity.png", "شکل ۴ — حساسیت برچسب به scaler، نمایش و تعداد مؤلفه")

    report.heading("۱۱. Registry، نسخه‌بندی و drift", 1)
    report.paragraph(
        "مدل assigner، centroidها، scaler، reducer و مدل‌های مکمل با metadata، تاریخ برازش، پارامترها، metric scoreboard و SHA-256 "
        "ثبت شدند. baseline drift برای ده PC نخست با PSI و KS روی دو نیمه ثابت داده ساخته شد. PSI>0.2 یا KS>0.1 آستانه هشدار "
        "است؛ این قرارداد می‌تواند در داده سال‌های آینده با پنجره زمانی واقعی جایگزین شود."
    )
    drift_display = drift.sort_values(["alert", "psi"], ascending=[False, False]).head(10)
    report.table(
        ["ویژگی", "PSI", "KS", "p-value", "هشدار"],
        [
            (row.feature, f"{row.psi:.4f}", f"{row.ks_statistic:.4f}", f"{row.ks_pvalue:.4g}", row.alert)
            for row in drift_display.itertuples()
        ],
    )

    report.heading("۱۲. مسیر دوم: NMF پربعد", 1)
    report.paragraph(
        "برای مسیر پیشرفته دوم، NMF مستقیما روی ۲۴۰ شاخص نامنفی فناوری اجرا شد. ویژگی breadth مقیاس‌شده کنار گذاشته شد تا "
        "فرض نامنفی‌بودن نقض نشود. انتخاب مؤلفه با silhouette کسینوسی در فضای فناوری، silhouette نهفته، کمترین سهم خوشه و "
        "reconstruction error انجام شد."
    )
    report.table(
        ["مؤلفه", "Silhouette کسینوسی", "نهفته", "کمترین سهم", "ARI", "انتخاب"],
        [
            (
                row.components,
                f"{row.technology_cosine_silhouette:.3f}",
                f"{row.latent_silhouette:.3f}",
                f"{100 * row.minimum_cluster_fraction:.1f}٪",
                f"{row.ari_vs_consensus:.3f}",
                "بله" if row.selected else "خیر",
            )
            for row in nmf.itertuples()
        ],
    )
    report.paragraph(
        f"راه‌حل {bonus['nmf']['selected_components']} مؤلفه با silhouette کسینوسی {bonus['nmf']['technology_cosine_silhouette']:.3f} "
        f"انتخاب شد. ARI={bonus['nmf']['ari_vs_consensus']:.3f} نسبت به consensus نشان می‌دهد NMF اکوسیستم فناوری را می‌یابد، "
        "در حالی که consensus عمدتا breadth را خلاصه می‌کند."
    )
    report.figure(figures / "bonus_nmf_comparison.png", "شکل ۵ — مقایسه تعداد مؤلفه‌ها در مسیر NMF")

    report.heading("۱۳. استنباط آماری و پایداری split", 1)
    report.heading("۱۳.۱. Bootstrap و permutation", 2)
    silhouette_ci = confidence.loc[confidence["metric"] == "silhouette"]
    report.table(
        ["روش", "برآورد", "کران پایین", "کران بالا"],
        [
            (row.method, f"{row.estimate:.4f}", f"{row.ci_lower:.4f}", f"{row.ci_upper:.4f}")
            for row in silhouette_ci.itertuples()
        ],
    )
    report.paragraph(
        f"برای پنج معیار، {bonus['significance']['bootstrap_repeats']} بازنمونه‌گیری و برای مقایسه دو روش برتر "
        f"{bonus['significance']['permutation_repeats']:,} جایگشت انجام شد. اختلاف silhouette روش "
        f"{bonus['significance']['best_method']} نسبت به {bonus['significance']['runner_up']} برابر "
        f"{bonus['significance']['mean_silhouette_difference']:.4f} با p={bonus['significance']['permutation_pvalue']:.4g} بود."
    )
    report.table(
        ["مقایسه", "روش اول", "روش دوم", "اختلاف", "p"],
        [
            (row.comparison, row.first_method, row.second_method, f"{row.mean_silhouette_difference:.4f}", f"{row.permutation_pvalue_two_sided:.4g}")
            for row in permutation.itertuples()
        ],
    )
    report.figure(figures / "bonus_silhouette_confidence_intervals.png", "شکل ۶ — فاصله اطمینان bootstrap برای silhouette")
    report.heading("۱۳.۲. پایداری بین دو نیمه تجربه", 2)
    report.paragraph(
        f"داده در ExperienceConsensus={bonus['split_stability']['threshold']:.0f} تقسیم و هر نیمه مستقل خوشه‌بندی شد. پس از تطبیق "
        f"prototypeها با Hungarian، میانگین Jaccard بیست فناوری شاخص {bonus['split_stability']['mean_top20_technology_jaccard']:.3f} "
        "بود. این نتیجه پایداری متوسط را تأیید می‌کند و هم‌زمان تفاوت میان گروه‌های تجربه را حفظ می‌کند."
    )
    report.table(
        ["خوشه اول", "خوشه دوم", "ردیف اول", "ردیف دوم", "Jaccard"],
        [
            (row.cluster_a, row.cluster_b, row.rows_a, row.rows_b, f"{row.top20_technology_jaccard:.3f}")
            for row in split.itertuples()
        ],
    )
    report.figure(figures / "bonus_split_stability.png", "شکل ۷ — تطبیق prototype میان دو نیمه تجربه")

    report.heading("۱۴. داشبورد و خروجی عملیاتی", 1)
    report.paragraph(
        "داشبورد Streamlit پنج صفحه دارد: Overview، Cluster Explorer، Evaluation، Live Assignment و 3D Explorer. صفحه انتساب "
        "رکورد یا CSV دارای PC1 تا PC50 را می‌پذیرد و برچسب و فاصله تا هر centroid را برمی‌گرداند. UMAP سه‌بعدی تعاملی روی "
        f"{bonus['umap_3d']['sample_size']:,} رکورد در داشبورد و فایل HTML مستقل ذخیره شده است. هیچ فاصله‌ای در UMAP برای برازش "
        "خوشه نهایی استفاده نشده و نمایش صرفا اکتشافی است."
    )

    report.heading("۱۵. محدودیت‌ها و نتیجه", 1)
    report.bullet("Silhouette متوسط و boundaryهای منفی نشان می‌دهند دو پروفایل overlap دارند.")
    report.bullet("ARI حساسیت به scaler و کاهش بعد، ادعای ذاتی‌بودن خوشه‌ها را تضعیف می‌کند.")
    report.bullet("داده خوداظهاری، داوطلبانه و تک‌ساله است؛ تعمیم زمانی یا جمعیتی قطعی نیست.")
    report.bullet("مدل پایین‌دستی جبران خدمت بهبود نداشت و کاربرد پیش‌بینی اقتصادی تأیید نشد.")
    report.bullet("SHAP روی PCها غیرمستقیم است و باید همراه فناوری‌ها و loadings خوانده شود.")
    report.paragraph(
        "Consensus یک تقسیم قابل‌توضیح و بازتولیدپذیر میان پشته متمرکزتر و breadth گسترده‌تر فراهم کرد، اما کاربرد آن در اکتشاف، "
        "خلاصه‌سازی و تولید فرضیه است. مسیر NMF، استنباط bootstrap/permutation، پایداری split و UMAP سه‌بعدی تصویر مکملی از "
        "ساختار ارائه می‌کنند و مانع تقلیل کل داده به یک پاسخ ساده می‌شوند."
    )

    report.heading("منابع", 1)
    report.paragraph(
        "منبع داده: Stack Overflow Developer Survey 2024. منابع روش شامل Scikit-learn، UMAP، Consensus Clustering، NMF و SHAP "
        "هستند. تمام محاسبات و اعداد از کد و خروجی‌های همین پروژه تولید شده‌اند."
    )
    report.code_block(".\\venv\\Scripts\\python.exe -m scripts.run_phase3 --config config.yaml")

    output = ROOT / "reports" / "Phase3_Report_FA.docx"
    report.save(output)
    return output


if __name__ == "__main__":
    print(build())
